#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <optional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "enn/trainer.hpp"
#include "enn/calibrator.hpp"

using namespace enn;

// Simple CSV reader for BICEP parquet data (converted to CSV)
struct StepFeature {
    F mean = 0.0;
    F std = 0.0;
    F q10 = 0.0;
    F q90 = 0.0;
    F aleatoric = 0.0;
    F epistemic = 0.0;
    F neff = 1.0; // Effective sample size (pseudo-count)
};

struct TrajectoryData {
    std::vector<std::vector<Vec>> sequences;
    std::vector<std::vector<F>> targets;          // scalar targets (output_dim=1)
    std::vector<std::vector<Vec>> multi_targets;  // multi-bit targets (output_dim>1)
    std::vector<std::vector<F>> weights;          // weights for loss (populated with neff)
    std::vector<std::vector<StepFeature>> features;
    std::vector<uint64_t> sequence_ids;
    int output_dim = 1;  // detected from CSV (1 = scalar target, >1 = multi-bit)
};

struct CliOptions {
    std::string csv_path;
    std::string telemetry_path = "enn_predictions.csv";
    std::optional<std::string> calibrator_path;
    std::optional<std::string> metadata_path;
    std::optional<std::string> sidecar_path;  // BICEP sidecar for verification binding
    bool predict_final_only = false;  // Grokking experiment: only score final timestep
    bool use_bce_loss = false;        // Use BCE loss instead of MSE (for binary classification)
    int train_split_mod = 5;          // h % train_split_mod == 0 goes to test set (~20% holdout)
    int output_dim = 0;               // 0 = auto-detect from CSV, >0 = override
    bool require_sidecar = false;     // If true, refuse to run without valid sidecar

    // Stability controls (fix optimization collapse before chasing grokking)
    F grad_clip_norm = 1.0;           // Clip global gradient norm (prevents collapse)
    F weight_decay = 1e-2;            // AdamW weight decay (the ONE regularization knob)
    int num_epochs = 2000;            // Number of training epochs

    // Checkpoint and early stopping
    std::optional<std::string> save_best_ckpt;  // Path to save best checkpoint
    int early_stop_patience = 50;               // Stop after N evals with no improvement
    int eval_every = 10;                        // Evaluate every N epochs
};

namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime  = 1099511628211ULL;

uint64_t fnv1a_bytes(const uint8_t* data, size_t n) {
    uint64_t h = kFnvOffset;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= kFnvPrime;
    }
    return h;
}

uint64_t hash_value(F value) {
    return fnv1a_bytes(reinterpret_cast<const uint8_t*>(&value), sizeof(F));
}

uint64_t hash_sequence_inputs(const std::vector<Vec>& seq) {
    uint64_t h = kFnvOffset;
    for (const auto& vec : seq) {
        for (int j = 0; j < vec.size(); ++j) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&vec[j]);
            for (size_t b = 0; b < sizeof(F); ++b) {
                h ^= bytes[b];
                h *= kFnvPrime;
            }
        }
    }
    return h;
}

struct FeatureSummary {
    std::vector<double> sum;
    std::vector<double> sum_sq;
    size_t count = 0;
};

void update_summary(FeatureSummary& summary, const Vec& vec) {
    if (summary.sum.empty()) {
        summary.sum.resize(vec.size(), 0.0);
        summary.sum_sq.resize(vec.size(), 0.0);
    }
    for (int d = 0; d < vec.size(); ++d) {
        double val = static_cast<double>(vec[d]);
        summary.sum[d] += val;
        summary.sum_sq[d] += val * val;
    }
    summary.count += 1;
}

void log_dataset_stats(const TrajectoryData& data) {
    if (data.sequences.empty()) {
        std::cout << "[Debug] No sequences loaded" << std::endl;
        return;
    }

    FeatureSummary summary;
    std::unordered_set<uint64_t> seq_hashes;
    std::vector<uint64_t> sample_hashes;

    for (size_t i = 0; i < data.sequences.size(); ++i) {
        const auto& seq = data.sequences[i];
        uint64_t h = hash_sequence_inputs(seq);
        seq_hashes.insert(h);
        if (sample_hashes.size() < 3) {
            sample_hashes.push_back(h);
        }
        for (const auto& vec : seq) {
            update_summary(summary, vec);
        }
    }

    std::map<F, size_t> label_counts;
    double label_sum = 0.0;
    size_t label_total = 0;
    for (const auto& tgt_seq : data.targets) {
        if (tgt_seq.empty()) continue;
        F y = tgt_seq.back();
        label_counts[y]++;
        label_sum += y;
        label_total++;
    }

    std::cout << "[Debug] sequences=" << data.sequences.size()
              << " unique_input_hashes=" << seq_hashes.size() << std::endl;
    if (!sample_hashes.empty()) {
        std::cout << "[Debug] sample input hashes:";
        for (auto h : sample_hashes) {
            std::cout << " 0x" << std::hex << h << std::dec;
        }
        std::cout << std::endl;
    }

    if (summary.count > 0) {
        std::cout << "[Debug] feature stats (mean ± std):" << std::endl;
        for (size_t d = 0; d < summary.sum.size(); ++d) {
            double mean = summary.sum[d] / summary.count;
            double var = summary.sum_sq[d] / summary.count - mean * mean;
            var = std::max(var, 0.0);
            std::cout << "  dim " << d << ": " << mean << " ± " << std::sqrt(var) << std::endl;
        }
    }

    if (label_total > 0) {
        std::cout << "[Debug] label mean=" << label_sum / label_total
                  << " counts:";
        for (const auto& kv : label_counts) {
            std::cout << " [" << kv.first << " -> " << kv.second << "]";
        }
        std::cout << std::endl;
    }
}

// =============================================================================
// SIDECAR VALIDATION - ENN only consumes features bound to BICEP checkpoint
// =============================================================================

std::string compute_file_sha256(const std::string& path) {
    // Use sha256sum command to compute SHA256 hash (matches Python's hashlib.sha256)
    std::string cmd = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Cannot run sha256sum for file: " + path);
    }

    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    int status = pclose(pipe);

    if (status != 0 || result.empty()) {
        throw std::runtime_error("sha256sum failed for file: " + path);
    }

    // sha256sum output format: "hash  filename\n" - extract first 64 chars
    if (result.length() < 64) {
        throw std::runtime_error("Invalid sha256sum output for file: " + path);
    }

    return result.substr(0, 64);
}

struct SidecarInfo {
    std::string features_shard_hash;
    std::string manifest_hash;
    std::string head_at_end;
    int step_start = 0;
    int step_end = 0;
    bool valid = false;
};

SidecarInfo load_sidecar(const std::string& path) {
    SidecarInfo info;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Sidecar] Cannot open sidecar file: " << path << std::endl;
        return info;
    }

    // Simple JSON parsing (minimal, no external deps)
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Extract key fields with simple string search
    auto extract_string = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return "";
        pos = content.find('"', pos + search.length());
        if (pos == std::string::npos) return "";
        size_t end = content.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return content.substr(pos + 1, end - pos - 1);
    };

    auto extract_int = [&](const std::string& key) -> int {
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.length();
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) ++pos;
        int val = 0;
        bool neg = false;
        if (pos < content.size() && content[pos] == '-') { neg = true; ++pos; }
        while (pos < content.size() && std::isdigit(content[pos])) {
            val = val * 10 + (content[pos] - '0');
            ++pos;
        }
        return neg ? -val : val;
    };

    info.features_shard_hash = extract_string("features_shard_hash");
    info.manifest_hash = extract_string("manifest_hash");
    info.head_at_end = extract_string("head_at_end");
    info.step_start = extract_int("step_start");
    info.step_end = extract_int("step_end");
    info.valid = !info.features_shard_hash.empty();

    return info;
}

bool validate_sidecar(const std::string& csv_path, const std::string& sidecar_path) {
    std::cout << "[Sidecar] Validating features binding..." << std::endl;

    SidecarInfo sidecar = load_sidecar(sidecar_path);
    if (!sidecar.valid) {
        std::cerr << "[Sidecar] ERROR: Invalid or missing sidecar" << std::endl;
        return false;
    }

    std::string actual_hash = compute_file_sha256(csv_path);

    std::cout << "[Sidecar] Binding info:" << std::endl;
    std::cout << "  manifest_hash: " << sidecar.manifest_hash << std::endl;
    std::cout << "  head_at_end: " << sidecar.head_at_end << std::endl;
    std::cout << "  step_range: [" << sidecar.step_start << ", " << sidecar.step_end << ")" << std::endl;
    std::cout << "  expected_hash: " << sidecar.features_shard_hash.substr(0, 16) << "..." << std::endl;
    std::cout << "  actual_hash:   " << actual_hash.substr(0, 16) << "..." << std::endl;

    // CRITICAL: Compare SHA256 hashes - reject if mismatch
    if (actual_hash != sidecar.features_shard_hash) {
        std::cerr << "[Sidecar] ERROR: Features hash mismatch!" << std::endl;
        std::cerr << "  Expected: " << sidecar.features_shard_hash << std::endl;
        std::cerr << "  Actual:   " << actual_hash << std::endl;
        std::cerr << "  REFUSING to process tampered/unbound features." << std::endl;
        return false;
    }

    std::cout << "[Sidecar] PASSED - Features hash verified, bound to BICEP checkpoint" << std::endl;
    return true;
}

void print_usage(const char* exe) {
    std::cerr << "Usage: " << exe << " <bicep_csv_file> [options]\n"
              << "\nOptions:\n"
              << "  --telemetry <path>     Output path for ENN predictions (default: enn_predictions.csv)\n"
              << "  --calibrator <path>    Path to Platt calibrator JSON\n"
              << "  --metadata <path>      Path to metadata JSON (validates std_ddof=1, quantile type7)\n"
              << "  --predict_final_only   Only compute loss at final timestep (for grokking experiments)\n"
              << "  --bce                  Use BCE loss instead of MSE (for binary classification)\n"
              << "  --train_split_mod <N>  Use hash%%N==0 for test split (default: 5 = ~20%% holdout)\n"
              << "\nVerification binding (BICEP -> ENN):\n"
              << "  --sidecar <path>       Path to BICEP features sidecar JSON\n"
              << "  --require_sidecar      REFUSE to run if sidecar missing/invalid (production mode)\n"
              << "\nStability controls (fix collapse before chasing grokking):\n"
              << "  --grad_clip <norm>     Clip global gradient norm (default: 1.0, 0=disabled)\n"
              << "  --weight_decay <wd>    AdamW weight decay (default: 0.01, the ONE reg knob)\n"
              << "  --epochs <N>           Number of training epochs (default: 2000)\n"
              << "\nCheckpoint and early stopping:\n"
              << "  --save_best_ckpt <path>      Path to save best checkpoint (e.g., /tmp/enn_best.ckpt)\n"
              << "  --early_stop_patience <N>    Stop after N evals with no improvement (default: 50)\n"
              << "  --eval_every <N>             Evaluate every N epochs (default: 10)\n"
              << std::endl;
}

void validate_metadata(const std::optional<std::string>& path) {
    if (!path) {
        return;
    }
    std::ifstream meta_file(*path);
    if (!meta_file.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + *path);
    }
    std::string json((std::istreambuf_iterator<char>(meta_file)), std::istreambuf_iterator<char>());
    std::string compact;
    compact.reserve(json.size());
    for (char ch : json) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            compact.push_back(ch);
        }
    }
    if (compact.find("\"std_ddof\":1") == std::string::npos) {
        throw std::runtime_error("Metadata must contain std_ddof = 1");
    }
    if (compact.find("type7") == std::string::npos) {
        throw std::runtime_error("Metadata must specify quantile_method type7");
    }
}

CliOptions parse_cli(int argc, char* argv[]) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--telemetry") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --telemetry");
            }
            opts.telemetry_path = argv[++i];
        } else if (arg == "--calibrator") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --calibrator");
            }
            opts.calibrator_path = std::string(argv[++i]);
        } else if (arg == "--metadata") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --metadata");
            }
            opts.metadata_path = std::string(argv[++i]);
        } else if (arg == "--predict_final_only") {
            opts.predict_final_only = true;
        } else if (arg == "--bce") {
            opts.use_bce_loss = true;
        } else if (arg == "--train_split_mod") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --train_split_mod");
            }
            opts.train_split_mod = std::stoi(argv[++i]);
        } else if (arg == "--grad_clip") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --grad_clip");
            }
            opts.grad_clip_norm = static_cast<F>(std::stod(argv[++i]));
        } else if (arg == "--weight_decay") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --weight_decay");
            }
            opts.weight_decay = static_cast<F>(std::stod(argv[++i]));
        } else if (arg == "--epochs") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --epochs");
            }
            opts.num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--save_best_ckpt") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --save_best_ckpt");
            }
            opts.save_best_ckpt = std::string(argv[++i]);
        } else if (arg == "--early_stop_patience") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --early_stop_patience");
            }
            opts.early_stop_patience = std::stoi(argv[++i]);
        } else if (arg == "--eval_every") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --eval_every");
            }
            opts.eval_every = std::stoi(argv[++i]);
        } else if (arg == "--output_dim") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --output_dim");
            }
            opts.output_dim = std::stoi(argv[++i]);
        } else if (arg == "--sidecar") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --sidecar");
            }
            opts.sidecar_path = std::string(argv[++i]);
        } else if (arg == "--require_sidecar") {
            opts.require_sidecar = true;
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        } else if (opts.csv_path.empty()) {
            opts.csv_path = arg;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + arg);
        }
    }

    if (opts.csv_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("CSV file path required");
    }

    return opts;
}

F sigmoid(F x) {
    if (x >= 0) {
        F z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    F z = std::exp(x);
    return z / (1.0f + z);
}

struct AlphaStats {
    F entropy = 0.0f;
    F alpha_max = 0.0f;
    int argmax = 0;
};

AlphaStats summarize_alpha(const Vec& alpha) {
    AlphaStats stats;
    for (int j = 0; j < alpha.size(); ++j) {
        F p = std::max(alpha[j], static_cast<F>(1e-9));
        stats.entropy -= p * std::log(p);
        if (p > stats.alpha_max) {
            stats.alpha_max = p;
            stats.argmax = j;
        }
    }
    return stats;
}

// Checkpoint saving utilities
void write_matrix_binary(std::ostream& os, const Mat& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    os.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    os.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    os.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(F));
}

void write_vector_binary(std::ostream& os, const Vec& v) {
    int size = static_cast<int>(v.size());
    os.write(reinterpret_cast<const char*>(&size), sizeof(int));
    os.write(reinterpret_cast<const char*>(v.data()), size * sizeof(F));
}

void write_scalar_binary(std::ostream& os, F value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(F));
}

bool save_checkpoint(const std::string& path, const SequenceTrainer& trainer,
                     int best_epoch, F best_test_acc) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open checkpoint file for writing: " << path << std::endl;
        return false;
    }

    // Write magic number and version for validation
    const uint32_t magic = 0x454E4E43; // "ENNC"
    const uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    // Write metadata
    out.write(reinterpret_cast<const char*>(&best_epoch), sizeof(int));
    write_scalar_binary(out, best_test_acc);

    // Write cell parameters
    const auto& cell = trainer.get_cell();
    write_matrix_binary(out, cell.Wx);
    write_matrix_binary(out, cell.Wh);
    write_matrix_binary(out, cell.L);
    write_vector_binary(out, cell.b);
    write_vector_binary(out, cell.ln_gamma);
    write_vector_binary(out, cell.ln_beta);
    write_scalar_binary(out, cell.log_lambda);

    // Write collapse parameters
    const auto& collapse = trainer.get_collapse();
    write_matrix_binary(out, collapse.Wq);
    write_matrix_binary(out, collapse.Wout);  // [output_dim x k] matrix
    write_vector_binary(out, collapse.bout);  // [output_dim] vector
    write_scalar_binary(out, collapse.log_temp);

    out.close();
    return out.good();
}

bool save_best_epoch_json(const std::string& ckpt_path, int best_epoch, F best_test_acc,
                          const CliOptions& options, const TrainConfig& config) {
    // Replace .ckpt with .json or append .json
    std::string json_path = ckpt_path;
    size_t ext_pos = json_path.rfind(".ckpt");
    if (ext_pos != std::string::npos) {
        json_path = json_path.substr(0, ext_pos) + "_epoch.json";
    } else {
        json_path += "_epoch.json";
    }

    std::ofstream out(json_path);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open JSON file for writing: " << json_path << std::endl;
        return false;
    }

    out << "{\n";
    out << "  \"best_epoch\": " << best_epoch << ",\n";
    out << "  \"best_test_acc\": " << std::fixed << std::setprecision(6) << best_test_acc << ",\n";
    out << "  \"config\": {\n";
    out << "    \"csv_path\": \"" << options.csv_path << "\",\n";
    out << "    \"num_epochs\": " << options.num_epochs << ",\n";
    out << "    \"eval_every\": " << options.eval_every << ",\n";
    out << "    \"early_stop_patience\": " << options.early_stop_patience << ",\n";
    out << "    \"learning_rate\": " << std::scientific << config.learning_rate << ",\n";
    out << "    \"weight_decay\": " << config.weight_decay << ",\n";
    out << "    \"grad_clip_norm\": " << std::fixed << config.grad_clip_norm << ",\n";
    out << "    \"batch_size\": " << config.batch_size << ",\n";
    out << "    \"loss_final_only\": " << (config.loss_final_only ? "true" : "false") << ",\n";
    out << "    \"use_bce_loss\": " << (config.use_bce_loss ? "true" : "false") << ",\n";
    out << "    \"train_split_mod\": " << options.train_split_mod << "\n";
    out << "  }\n";
    out << "}\n";

    out.close();
    return out.good();
}

} // namespace

TrajectoryData load_bicep_data(const std::string& csv_file) {
    TrajectoryData data;
    std::ifstream file(csv_file);
    std::string line;
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + csv_file);
    }
    
    // Read header and build column index map
    std::getline(file, line);
    // Strip trailing carriage return for Windows-style CRLF line endings
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    std::vector<std::string> header;
    {
        std::istringstream header_stream(line);
        std::string htok;
        while (std::getline(header_stream, htok, ',')) {
            header.push_back(htok);
        }
    }

    auto col_index = [&](const std::string& name) -> size_t {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) {
            throw std::runtime_error("Missing column in CSV: " + name);
        }
        return static_cast<size_t>(std::distance(header.begin(), it));
    };

    auto col_if_present = [&](const std::string& name) -> std::optional<size_t> {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) {
            return std::nullopt;
        }
        return static_cast<size_t>(std::distance(header.begin(), it));
    };

    const size_t idx_sequence_id = col_index("sequence_id");
    const size_t idx_step = col_index("step");
    const auto idx_state0_opt = col_if_present("state_0");
    const auto idx_state_col = col_if_present("state");
    const auto idx_input_opt = col_if_present("input");
    const size_t idx_input = idx_input_opt
        ? *idx_input_opt
        : (idx_state0_opt ? *idx_state0_opt : col_index("state_0"));

    // Detect multi-bit targets: look for target_bit_0, target_bit_1, ... columns
    std::vector<size_t> idx_target_bits;
    for (int bit = 0; ; ++bit) {
        auto idx_bit = col_if_present("target_bit_" + std::to_string(bit));
        if (!idx_bit) break;
        idx_target_bits.push_back(*idx_bit);
    }
    const bool is_multi_bit = !idx_target_bits.empty();
    data.output_dim = is_multi_bit ? static_cast<int>(idx_target_bits.size()) : 1;

    // For scalar target, require "target" column
    std::optional<size_t> idx_target_opt;
    if (!is_multi_bit) {
        idx_target_opt = col_index("target");
    }

    const auto idx_state_mean_col = col_if_present("state_mean");
    const auto idx_state_std_col = col_if_present("state_std");
    const auto idx_state_q10_col = col_if_present("state_q10");
    const auto idx_state_q90_col = col_if_present("state_q90");
    const auto idx_aleatoric_col = col_if_present("aleatoric_unc");
    const auto idx_epistemic_col = col_if_present("epistemic_unc");
    const auto idx_weight_col = col_if_present("weight");

    auto resolve_with_fallback = [&](const std::optional<size_t>& primary,
                                     const std::optional<size_t>& secondary,
                                     size_t fallback) -> size_t {
        if (primary) return *primary;
        if (secondary) return *secondary;
        return fallback;
    };

    const size_t idx_state_mean = idx_state_mean_col
        ? *idx_state_mean_col
        : resolve_with_fallback(idx_state0_opt, idx_state_col, idx_input);
    const size_t idx_state_std = idx_state_std_col
        ? *idx_state_std_col
        : idx_state_mean;
    const size_t idx_state_q10 = idx_state_q10_col
        ? *idx_state_q10_col
        : idx_state_mean;
    const size_t idx_state_q90 = idx_state_q90_col
        ? *idx_state_q90_col
        : idx_state_mean;
    const size_t idx_aleatoric = idx_aleatoric_col
        ? *idx_aleatoric_col
        : idx_state_mean;
    const size_t idx_epistemic = idx_epistemic_col
        ? *idx_epistemic_col
        : idx_state_mean;
    
    std::map<uint64_t, std::vector<Vec>> sequence_map;
    std::map<uint64_t, std::vector<F>> target_map;          // scalar targets
    std::map<uint64_t, std::vector<Vec>> multi_target_map;  // multi-bit targets
    std::map<uint64_t, std::vector<F>> weight_map;          // weights
    std::map<uint64_t, std::vector<StepFeature>> feature_map;
    std::unordered_map<uint64_t, int> last_step_seen;

    while (std::getline(file, line)) {
        // Strip trailing carriage return for Windows-style CRLF line endings
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Check we have required columns based on mode
        size_t min_cols = std::max(idx_sequence_id, idx_step) + 1;
        if (is_multi_bit) {
            for (size_t idx : idx_target_bits) {
                min_cols = std::max(min_cols, idx + 1);
            }
        } else {
            min_cols = std::max(min_cols, *idx_target_opt + 1);
        }
        if (tokens.size() < min_cols) continue;

        auto parse_value = [&](size_t idx) -> F {
            if (idx >= tokens.size() || tokens[idx].empty()) {
                return 0.0;
            }
            return static_cast<F>(std::stod(tokens[idx]));
        };

        uint64_t sequence_id = std::stoull(tokens[idx_sequence_id]);
        uint32_t step = static_cast<uint32_t>(std::stoul(tokens[idx_step]));
        F input = parse_value(idx_input);
        F state_mean = parse_value(idx_state_mean);
        F state_std = parse_value(idx_state_std);
        F state_q10 = parse_value(idx_state_q10);
        F state_q90 = parse_value(idx_state_q90);
        F aleatoric = parse_value(idx_aleatoric);
        F epistemic = parse_value(idx_epistemic);
        
        // Weight from CSV is assumed to be 1/variance (inverse variance)
        F raw_inv_var = idx_weight_col ? parse_value(*idx_weight_col) : 1.0;

        // Parse targets based on mode
        F scalar_target = 0.0;
        Vec multi_target;
        if (is_multi_bit) {
            multi_target.resize(static_cast<int>(idx_target_bits.size()));
            for (size_t b = 0; b < idx_target_bits.size(); ++b) {
                F bit_val = parse_value(idx_target_bits[b]);
                if (!(bit_val >= -1e-6 && bit_val <= 1.0 + 1e-6)) {
                    throw std::runtime_error("target_bit outside [0,1]");
                }
                multi_target(static_cast<int>(b)) = bit_val;
            }
        } else {
            scalar_target = parse_value(*idx_target_opt);
            if (!(0.0 - 1e-6 <= scalar_target && scalar_target <= 1.0 + 1e-6)) {
                throw std::runtime_error("target outside [0,1]");
            }
        }

        // Compute Effective Sample Size (neff) = q(1-q) / variance
        // variance = 1 / raw_inv_var
        // neff = q(1-q) * raw_inv_var
        // Clamp q to avoid 0 variance at extrema
        F q_safe = std::max(1e-6, std::min(1.0 - 1e-6, scalar_target));
        F bernoulli_var = q_safe * (1.0 - q_safe);
        F neff = bernoulli_var * raw_inv_var;
        
        // Clamp neff to avoid explosion and extremely high weights
        neff = std::max(1.0, std::min(1e6, neff));

        // Invariants for features
        if (state_std < 0 || aleatoric < 0 || epistemic < 0) {
            throw std::runtime_error("Negative variance/uncertainty encountered");
        }
        if (state_q10 > state_q90) {
            throw std::runtime_error("state_q10 greater than state_q90");
        }
        if (!(state_q10 - 1e-6 <= state_mean && state_mean <= state_q90 + 1e-6)) {
            throw std::runtime_error("state_mean outside [q10,q90]");
        }
        int expected_step = 0;
        auto it_step = last_step_seen.find(sequence_id);
        if (it_step != last_step_seen.end()) {
            expected_step = it_step->second + 1;
        }
        if (static_cast<int>(step) != expected_step) {
            throw std::runtime_error("Non-consecutive step for sequence " + std::to_string(sequence_id));
        }
        last_step_seen[sequence_id] = step;

        // Build feature vector [base input, mean, std, q10, q90, aleatoric, epistemic, neff]
        const int feature_dim = 8; // Added neff
        Vec input_vec(feature_dim);
        input_vec << input, state_mean, state_std, state_q10, state_q90, aleatoric, epistemic, neff;

        if (sequence_map.find(sequence_id) == sequence_map.end()) {
            sequence_map[sequence_id] = std::vector<Vec>();
            if (is_multi_bit) {
                multi_target_map[sequence_id] = std::vector<Vec>();
            } else {
                target_map[sequence_id] = std::vector<F>();
            }
            weight_map[sequence_id] = std::vector<F>();
        }

        // Ensure vectors are large enough
        if (sequence_map[sequence_id].size() <= step) {
            sequence_map[sequence_id].resize(step + 1);
            if (is_multi_bit) {
                multi_target_map[sequence_id].resize(step + 1);
            } else {
                target_map[sequence_id].resize(step + 1);
            }
            weight_map[sequence_id].resize(step + 1);
        }

        sequence_map[sequence_id][step] = input_vec;
        if (is_multi_bit) {
            multi_target_map[sequence_id][step] = multi_target;
        } else {
            target_map[sequence_id][step] = scalar_target;
        }
        weight_map[sequence_id][step] = neff; // Use neff as weight

        auto& feature_seq = feature_map[sequence_id];
        if (feature_seq.size() <= step) {
            feature_seq.resize(step + 1);
        }

        StepFeature feat;
        feat.mean = state_mean;
        feat.std = state_std;
        feat.q10 = state_q10;
        feat.q90 = state_q90;
        feat.aleatoric = aleatoric;
        feat.epistemic = epistemic;
        feat.neff = neff;
        feature_seq[step] = feat;
    }
    
    // Convert map to vectors
    for (const auto& pair : sequence_map) {
        auto seq_id = pair.first;
        data.sequences.push_back(pair.second);
        if (is_multi_bit) {
            data.multi_targets.push_back(multi_target_map[seq_id]);
        } else {
            data.targets.push_back(target_map[seq_id]);
        }
        data.weights.push_back(weight_map[seq_id]);
        data.features.push_back(feature_map[seq_id]);
        data.sequence_ids.push_back(seq_id);
    }

    if (is_multi_bit) {
        std::cout << "[Multi-bit mode] Detected " << data.output_dim << " target bits" << std::endl;
    }

    return data;
}

int main(int argc, char* argv[]) {
    CliOptions options;
    try {
        options = parse_cli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    try {
        validate_metadata(options.metadata_path);
    } catch (const std::exception& e) {
        std::cerr << "Metadata validation failed: " << e.what() << std::endl;
        return 1;
    }

    // ==========================================================================
    // SIDECAR VALIDATION - ENN only consumes features bound to BICEP checkpoint
    // This is the spine of the whole product. In production mode, refuse to run
    // if sidecar is missing or invalid.
    // ==========================================================================
    if (options.sidecar_path) {
        bool valid = validate_sidecar(options.csv_path, *options.sidecar_path);
        if (!valid && options.require_sidecar) {
            std::cerr << "[FATAL] Sidecar validation failed and --require_sidecar is set" << std::endl;
            std::cerr << "        ENN requires features to be bound to a BICEP checkpoint receipt." << std::endl;
            return 1;
        }
    } else if (options.require_sidecar) {
        std::cerr << "[FATAL] --require_sidecar set but no --sidecar provided" << std::endl;
        std::cerr << "        ENN requires features to be bound to a BICEP checkpoint receipt." << std::endl;
        std::cerr << "        Run BICEP with trace emission enabled to generate sidecar." << std::endl;
        return 1;
    } else {
        std::cout << "[Sidecar] No sidecar provided (use --sidecar for verification binding)" << std::endl;
    }

    std::string csv_file = options.csv_path;
    std::cout << "=== BICEP -> ENN-C++ Integration ===" << std::endl;
    
    try {
        // Load BICEP trajectory data
        std::cout << "Loading BICEP trajectory data from: " << csv_file << std::endl;
        TrajectoryData traj_data = load_bicep_data(csv_file);
        log_dataset_stats(traj_data);
        
        std::cout << "Loaded " << traj_data.sequences.size() << " sequences" << std::endl;
        if (traj_data.sequences.empty()) {
            std::cerr << "No data loaded!" << std::endl;
            return 1;
        }
        
        // Print sample data
        const bool is_multi_bit = traj_data.output_dim > 1;
        std::cout << "Sample sequence (first 5 steps):" << std::endl;
        for (size_t i = 0; i < std::min(5UL, traj_data.sequences[0].size()); ++i) {
            std::cout << "  Step " << i << ": features=" << traj_data.sequences[0][i].transpose();
            if (is_multi_bit) {
                std::cout << ", target_bits=" << traj_data.multi_targets[0][i].transpose();
            } else {
                std::cout << ", target=" << traj_data.targets[0][i];
            }
            std::cout << std::endl;
        }

        // Determine output_dim from data or CLI override
        const int output_dim = (options.output_dim > 0) ? options.output_dim : traj_data.output_dim;

        // Convert to ENN SeqBatch format with deterministic train/test split
        // Split uses hash of sequence inputs: h % train_split_mod == 0 -> test
        SeqBatch train_data, test_data;
        train_data.output_dim = output_dim;
        test_data.output_dim = output_dim;
        std::vector<size_t> train_indices, test_indices;
        bool has_weights = !traj_data.weights.empty() && !traj_data.weights[0].empty();

        for (size_t i = 0; i < traj_data.sequences.size(); ++i) {
            uint64_t h = hash_sequence_inputs(traj_data.sequences[i]);
            bool is_test_seq = (h % static_cast<uint64_t>(options.train_split_mod) == 0);
            if (is_test_seq) {
                test_data.sequences.push_back(traj_data.sequences[i]);
                if (is_multi_bit) {
                    test_data.multi_targets.push_back(traj_data.multi_targets[i]);
                } else {
                    test_data.targets.push_back(traj_data.targets[i]);
                }
                if (has_weights) test_data.weights.push_back(traj_data.weights[i]);
                test_indices.push_back(i);
            } else {
                train_data.sequences.push_back(traj_data.sequences[i]);
                if (is_multi_bit) {
                    train_data.multi_targets.push_back(traj_data.multi_targets[i]);
                } else {
                    train_data.targets.push_back(traj_data.targets[i]);
                }
                if (has_weights) train_data.weights.push_back(traj_data.weights[i]);
                train_indices.push_back(i);
            }
        }
        // Ensure at least one test sample exists
        if (test_data.batch_size() == 0 && train_data.batch_size() > 1) {
            test_data.sequences.push_back(train_data.sequences.back());
            if (is_multi_bit) {
                test_data.multi_targets.push_back(train_data.multi_targets.back());
            } else {
                test_data.targets.push_back(train_data.targets.back());
            }
            if (has_weights) test_data.weights.push_back(train_data.weights.back());
            test_indices.push_back(train_indices.back());
            train_data.sequences.pop_back();
            if (is_multi_bit) {
                train_data.multi_targets.pop_back();
            } else {
                train_data.targets.pop_back();
            }
            if (has_weights) train_data.weights.pop_back();
            train_indices.pop_back();
        }
        std::cout << "Split: " << train_data.batch_size() << " train, "
                  << test_data.batch_size() << " test (mod=" << options.train_split_mod << ")" << std::endl;

        // LEAKAGE DETECTION: Check for overlapping unique inputs between train/test
        {
            std::unordered_set<uint64_t> train_hashes, test_hashes;
            for (const auto& seq : train_data.sequences) {
                train_hashes.insert(hash_sequence_inputs(seq));
            }
            for (const auto& seq : test_data.sequences) {
                test_hashes.insert(hash_sequence_inputs(seq));
            }
            size_t intersection = 0;
            for (auto h : test_hashes) {
                if (train_hashes.count(h)) ++intersection;
            }
            std::cout << "[Leakage Check] train_unique=" << train_hashes.size()
                      << " test_unique=" << test_hashes.size()
                      << " intersection=" << intersection;
            if (intersection > 0) {
                std::cout << " ⚠️  WARNING: " << intersection << " test inputs appear in train set!";
            } else {
                std::cout << " ✓ disjoint";
            }
            std::cout << std::endl;
        }

        // Configure ENN trainer for parity task
        TrainConfig config;
        config.learning_rate = 1e-3;
        config.batch_size = 16;
        config.epochs = options.num_epochs;
        config.verbose = true;
        config.print_every = 10;
        config.loss_final_only = options.predict_final_only;  // Grokking experiment flag
        config.use_bce_loss = options.use_bce_loss || is_multi_bit;  // BCE loss for binary/multi-bit
        config.use_weighted_loss = has_weights; // Enable weighted loss if weights are present
        config.output_dim = output_dim;                        // Multi-bit output dimension

        // STABILITY CONTROLS (fix collapse before chasing grokking)
        // (A) Gradient clipping - prevents optimization collapse / grad spikes
        config.grad_clip_norm = options.grad_clip_norm;

        // (B) Only decay weight matrices, NOT biases or LayerNorm params
        config.decay_weights_only = true;

        // (C) Don't double-regularize: pick ONE of AdamW decay or explicit reg_eta
        config.use_adamw_decay = true;           // Use AdamW weight_decay as the regularization knob
        config.weight_decay = options.weight_decay;  // The ONE regularization knob
        config.reg_eta = 0.0;                    // Disable explicit L2 (would double-regularize)
        config.reg_beta = 0.0;                   // Disable PSD regularizer for now

        const int k = 32;
        const int feature_dim = 7;
        const int hidden_dim = 64;
        const F lambda = 0.05;
        config.frontend_filters = 32;
        config.frontend_temporal_kernel = 5;
        config.frontend_depth_kernel = 3;
        config.embed_dim = 32;
        config.use_layer_norm = true;

        std::cout << "\n=== Training ENN on BICEP Trajectories ===" << std::endl;
        std::cout << "Stability settings:" << std::endl;
        std::cout << "  grad_clip_norm=" << config.grad_clip_norm << std::endl;
        std::cout << "  weight_decay=" << config.weight_decay << " (AdamW, the ONE reg knob)" << std::endl;
        std::cout << "  decay_weights_only=" << (config.decay_weights_only ? "true" : "false") << std::endl;
        std::cout << "  loss_final_only=" << (config.loss_final_only ? "true" : "false") << std::endl;
        std::cout << "  use_bce_loss=" << (config.use_bce_loss ? "true" : "false") << std::endl;
        std::cout << "  epochs=" << config.epochs << std::endl;
        std::cout << "  eval_every=" << options.eval_every << std::endl;
        std::cout << "  early_stop_patience=" << options.early_stop_patience << std::endl;
        if (options.save_best_ckpt) {
            std::cout << "  save_best_ckpt=" << *options.save_best_ckpt << std::endl;
        }
        SequenceTrainer trainer(k, feature_dim, config.embed_dim, hidden_dim, lambda, config);

        // Open curves CSV for grokking analysis
        const std::string curves_path = options.telemetry_path + ".curves.csv";
        std::ofstream curves_out(curves_path);
        curves_out << "epoch,train_loss,train_eval_loss,train_acc,test_eval_loss,test_acc\n";

        // Training loop with train/test curve logging
        F best_loss = std::numeric_limits<F>::max();

        // Early stopping and best checkpoint tracking
        F best_test_acc = 0.0;
        int best_epoch = 0;
        int epochs_without_improvement = 0;
        bool early_stopped = false;

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            F train_loss = trainer.train_epoch(train_data);

            if (train_loss < best_loss) {
                best_loss = train_loss;
            }

            if (epoch % options.eval_every == 0) {
                // Evaluate on both train and test sets for grokking curves
                Metrics train_metrics, test_metrics;
                F train_eval = trainer.evaluate(train_data, train_metrics);
                F test_eval = trainer.evaluate(test_data, test_metrics);

                // Log to CSV for grokking analysis
                curves_out << epoch << "," << train_loss << "," << train_eval << ","
                           << train_metrics.accuracy << "," << test_eval << ","
                           << test_metrics.accuracy << "\n";
                curves_out.flush();

                std::cout << "Epoch " << std::setw(3) << epoch
                          << " | train_loss=" << std::fixed << std::setprecision(4) << train_loss
                          << " | train_eval=" << train_eval << " acc=" << std::setprecision(2) << train_metrics.accuracy
                          << " | test_eval=" << test_eval << " acc=" << test_metrics.accuracy;

                // Check for improvement in test accuracy
                if (test_metrics.accuracy > best_test_acc) {
                    best_test_acc = test_metrics.accuracy;
                    best_epoch = epoch;
                    epochs_without_improvement = 0;

                    std::cout << "\n>>> New best test acc: " << std::fixed << std::setprecision(2)
                              << (best_test_acc * 100.0) << "% at epoch " << best_epoch;

                    // Save checkpoint if path is specified
                    if (options.save_best_ckpt) {
                        if (save_checkpoint(*options.save_best_ckpt, trainer, best_epoch, best_test_acc)) {
                            save_best_epoch_json(*options.save_best_ckpt, best_epoch, best_test_acc, options, config);
                        }
                    }
                } else {
                    epochs_without_improvement++;
                }

                std::cout << std::endl;

                // Check for early stopping
                if (epochs_without_improvement >= options.early_stop_patience) {
                    std::cout << ">>> Early stopping at epoch " << epoch
                              << " (no improvement for " << epochs_without_improvement << " evals)" << std::endl;
                    early_stopped = true;
                    break;
                }
            }
        }
        curves_out.close();
        std::cout << "Curves saved to: " << curves_path << std::endl;

        std::cout << "\n=== Training Complete ===" << std::endl;
        std::cout << "Best test acc: " << std::fixed << std::setprecision(2)
                  << (best_test_acc * 100.0) << "% at epoch " << best_epoch << std::endl;
        if (options.save_best_ckpt) {
            std::cout << "Checkpoint saved to: " << *options.save_best_ckpt << std::endl;
        }
        if (early_stopped) {
            std::cout << "Training stopped early due to lack of improvement." << std::endl;
        }
        std::cout << "Best train loss: " << best_loss << std::endl;

        // Final evaluation on test set
        Metrics final_test_metrics;
        F final_test_loss = trainer.evaluate(test_data, final_test_metrics);
        std::cout << "Final test loss: " << final_test_loss
                  << " | acc: " << final_test_metrics.accuracy << std::endl;

        // Test on a few sequences from test set
        std::cout << "\nTesting on sample test sequences:" << std::endl;
        for (size_t i = 0; i < std::min(5UL, test_data.sequences.size()); ++i) {
            auto predictions = trainer.forward_sequence(test_data.sequences[i]);
            F final_pred = predictions.back();
            F target = test_data.targets[i].back();
            bool correct = (final_pred > 0.5) == (target > 0.5);

            std::cout << "TestSeq " << i << ": pred=" << std::setprecision(3) << final_pred
                      << ", target=" << target << ", correct=" << (correct ? "YES" : "NO") << std::endl;
        }
        
        // Save ENN predictions for FusionAlpha
        std::cout << "\n=== Saving ENN Outputs for FusionAlpha ===" << std::endl;

        Calibrator calibrator = options.calibrator_path
            ? Calibrator::from_json_file(*options.calibrator_path)
            : Calibrator::identity();
        std::cout << "Calibrator: " << calibrator.calibrator_id << std::endl;
        
        std::ofstream enn_output(options.telemetry_path);
        enn_output << "sequence_id,step,margin,q_pred,obs_reliability,alpha_entropy,alpha_max,attention_argmax,collapse_temperature,state_mean,state_std,state_q10,state_q90,aleatoric_unc,epistemic_unc,target,calibrator_id\n";

        double margin_sum = 0.0;
        double margin_sq = 0.0;
        size_t margin_count = 0;
        std::map<int, size_t> margin_hist;

        for (size_t i = 0; i < train_data.sequences.size(); ++i) {
            Vec final_alpha;
            F collapse_temp = 1.0;
            auto predictions = trainer.forward_sequence(
                train_data.sequences[i], nullptr, nullptr, &final_alpha, &collapse_temp);
            F margin = predictions.back();
            F q_pred = sigmoid(margin);
            F obs_reliability = calibrator.calibrate(margin);
            F target = train_data.targets[i].back();
            const StepFeature& feat = traj_data.features[i].back();
            AlphaStats stats = summarize_alpha(final_alpha);
            size_t final_step = train_data.sequences[i].empty() ? 0 : (train_data.sequences[i].size() - 1);

            enn_output << traj_data.sequence_ids[i] << ","
                       << final_step << ","
                       << margin << ","
                       << q_pred << ","
                       << obs_reliability << ","
                       << stats.entropy << ","
                       << stats.alpha_max << ","
                       << stats.argmax << ","
                       << collapse_temp << ","
                       << feat.mean << ","
                       << feat.std << ","
                       << feat.q10 << ","
                       << feat.q90 << ","
                       << feat.aleatoric << ","
                       << feat.epistemic << ","
                       << target << ","
                       << calibrator.calibrator_id << "\n";

            margin_sum += margin;
            margin_sq += margin * margin;
            margin_count += 1;
            int bucket = static_cast<int>(std::round(margin * 1000.0));
            margin_hist[bucket]++;
        }

        enn_output.close();
        if (margin_count > 0) {
            double mean = margin_sum / margin_count;
            double var = margin_sq / margin_count - mean * mean;
            if (var < 0.0) var = 0.0;
            std::cout << "[Debug] Margin mean=" << mean << " std=" << std::sqrt(var)
                      << " samples=" << margin_count << std::endl;
            std::cout << "[Debug] Margin histogram (scaled x1000):";
            for (const auto& kv : margin_hist) {
                std::cout << " [" << kv.first << " -> " << kv.second << "]";
            }
            std::cout << std::endl;
        }
        std::cout << "Saved ENN predictions to: " << options.telemetry_path << std::endl;
        
        std::cout << "\n✅ BICEP -> ENN-C++ pipeline completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

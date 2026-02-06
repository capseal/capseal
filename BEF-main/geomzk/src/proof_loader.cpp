#include "geomzk/proof_loader.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "geomzk/json.hpp"

namespace geomzk {
namespace {

Hash parse_hash(const std::string& hex) {
    if (hex.size() != 64) {
        throw std::runtime_error("expected 64 hex chars for hash");
    }
    Hash h;
    auto hex_value = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (std::size_t i = 0; i < 32; ++i) {
        int hi = hex_value(hex[2 * i]);
        int lo = hex_value(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("invalid hex digit");
        h.bytes[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return h;
}

RowBackendKind parse_backend(const std::string& name) {
    if (name == "merkle" || name == "geom_plain_fri") {
        return RowBackendKind::MERKLE;
    }
    if (name == "stc" || name == "geom_stc_fri") {
        return RowBackendKind::STC;
    }
    throw std::runtime_error("unsupported row backend: " + name);
}

Field parse_field_from_hex(const std::string& value) {
    if (value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0) {
        std::uint64_t acc = 0;
        for (std::size_t i = 2; i < value.size(); ++i) {
            char c = value[i];
            int digit = 0;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
            else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
            else throw std::runtime_error("invalid hex field element");
            acc = (acc << 4) | static_cast<std::uint64_t>(digit);
        }
        return Field(acc);
    }
    std::int64_t as_int = std::stoll(value);
    return Field(static_cast<std::uint64_t>(as_int));
}

Field parse_field(const JsonValue& node) {
    if (node.is_string()) {
        return parse_field_from_hex(node.as_string());
    }
    return Field(static_cast<std::uint64_t>(node.as_int()));
}

bool is_zero_hash(const Hash& h) {
    return std::all_of(h.bytes.begin(), h.bytes.end(), [](std::uint8_t b) { return b == 0; });
}

const JsonValue* find_member(const JsonValue& node, const char* key) {
    if (!node.is_object()) return nullptr;
    const auto& obj = node.as_object();
    auto it = obj.find(key);
    if (it == obj.end()) return nullptr;
    return &it->second;
}

} // namespace

Proof load_proof_from_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open proof file: " + path);
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    JsonValue root = JsonValue::parse(data);

    Proof proof;
    const auto& stmt_node = root.at("statement");
    if (const JsonValue* steps_node = find_member(stmt_node, "steps")) {
        proof.statement.steps = static_cast<std::uint32_t>(steps_node->as_int());
    } else if (const JsonValue* params_node = find_member(stmt_node, "params")) {
        if (const JsonValue* inner_steps = find_member(*params_node, "steps")) {
            proof.statement.steps = static_cast<std::uint32_t>(inner_steps->as_int());
        }
    }

    const auto& rc_node = root.at("row_commitment");
    const auto& rc_obj = rc_node.as_object();
    if (rc_obj.count("backend")) {
        proof.row_commitment.backend = parse_backend(rc_node.at("backend").as_string());
    } else if (rc_obj.count("chunk_len") || rc_obj.count("num_chunks")) {
        proof.row_commitment.backend = RowBackendKind::STC;
    }
    if (rc_obj.count("root")) {
        proof.row_commitment.root = parse_hash(rc_node.at("root").as_string());
    }
    if (rc_obj.count("row_size")) {
        proof.row_commitment.row_size = static_cast<std::uint32_t>(rc_node.at("row_size").as_int());
    }
    if (rc_obj.count("row_width")) {
        proof.row_commitment.row_size = static_cast<std::uint32_t>(rc_node.at("row_width").as_int());
    }
    if (rc_obj.count("n_rows")) {
        proof.row_commitment.n_rows = static_cast<std::uint32_t>(rc_node.at("n_rows").as_int());
    }
    if (rc_obj.count("chunk_len")) {
        proof.row_commitment.chunk_len = static_cast<std::uint32_t>(rc_node.at("chunk_len").as_int());
    }
    if (rc_obj.count("num_chunks")) {
        proof.row_commitment.num_chunks = static_cast<std::uint32_t>(rc_node.at("num_chunks").as_int());
    }
    if (rc_obj.count("chunk_tree_arity")) {
        proof.row_commitment.chunk_tree_arity = static_cast<std::uint32_t>(rc_node.at("chunk_tree_arity").as_int());
    }

    if (rc_obj.count("params")) {
        const auto& params = rc_node.at("params");
        if (is_zero_hash(proof.row_commitment.root) && params.as_object().count("root")) {
            proof.row_commitment.root = parse_hash(params.at("root").as_string());
        }
        if (params.as_object().count("length")) {
            std::uint32_t length = static_cast<std::uint32_t>(params.at("length").as_int());
            if (proof.row_commitment.row_size && !proof.row_commitment.n_rows) {
                proof.row_commitment.n_rows = length / proof.row_commitment.row_size;
            }
        }
        if (params.as_object().count("chunk_len")) {
            proof.row_commitment.chunk_len = static_cast<std::uint32_t>(params.at("chunk_len").as_int());
        }
        if (params.as_object().count("num_chunks")) {
            proof.row_commitment.num_chunks = static_cast<std::uint32_t>(params.at("num_chunks").as_int());
        }
        if (params.as_object().count("chunk_tree_arity")) {
            proof.row_commitment.chunk_tree_arity = static_cast<std::uint32_t>(params.at("chunk_tree_arity").as_int());
        }
    }

    if (!proof.row_commitment.n_rows && proof.statement.steps) {
        proof.row_commitment.n_rows = proof.statement.steps;
    }

    if (is_zero_hash(proof.row_commitment.root)) {
        // root may be provided as "global_root"
        if (rc_node.as_object().count("global_root")) {
            proof.row_commitment.root = parse_hash(rc_node.at("global_root").as_string());
        }
    }

    if (root.as_object().count("fri_params")) {
        const auto& fri_node = root.at("fri_params");
        proof.fri_params.domain_size = static_cast<std::uint32_t>(fri_node.at("domain_size").as_int());
        proof.fri_params.max_degree = static_cast<std::uint32_t>(fri_node.at("max_degree").as_int());
        proof.fri_params.num_rounds = static_cast<std::uint32_t>(fri_node.at("num_rounds").as_int());
        proof.fri_params.num_queries = static_cast<std::uint32_t>(fri_node.at("num_queries").as_int());
    }

    if (const JsonValue* fri_proof_node = find_member(root, "fri_proof")) {
        if (const JsonValue* layer_roots = find_member(*fri_proof_node, "layer_roots")) {
            const auto& layer_arr = layer_roots->as_array();
            for (const auto& item : layer_arr) {
                proof.fri_proof.layer_roots.push_back(parse_hash(item.as_string()));
            }
        } else if (const JsonValue* layers = find_member(*fri_proof_node, "layers")) {
            const auto& layer_arr = layers->as_array();
            for (const auto& layer : layer_arr) {
                const JsonValue* commit = find_member(layer, "commitment");
                if (commit) {
                    if (const JsonValue* root_field = find_member(*commit, "root")) {
                        proof.fri_proof.layer_roots.push_back(parse_hash(root_field->as_string()));
                    }
                } else if (const JsonValue* root_field = find_member(layer, "root")) {
                    proof.fri_proof.layer_roots.push_back(parse_hash(root_field->as_string()));
                }
            }
        }

        if (const JsonValue* queries_node = find_member(*fri_proof_node, "queries")) {
            const auto& query_arr = queries_node->as_array();
            for (const auto& qnode : query_arr) {
                FriQuery q;
                if (const JsonValue* base_idx = find_member(qnode, "base_index")) {
                    q.base_index = static_cast<std::uint32_t>(base_idx->as_int());
                }
                proof.fri_proof.queries.push_back(q);
            }
        }
    }

    const auto& openings = root.at("row_openings").as_array();
    auto parse_stc_chunk = [&](const JsonValue& chunk_obj, std::uint32_t row_index) {
        RowOpening opening;
        opening.row_index = row_index;
        const auto& values = chunk_obj.at("values").as_array();
        opening.row_values.reserve(values.size());
        for (const auto& val : values) {
            opening.row_values.push_back(parse_field(val));
        }
        opening.chunk_index = static_cast<std::uint32_t>(chunk_obj.at("chunk_index").as_int());
        opening.chunk_offset = static_cast<std::uint32_t>(chunk_obj.at("chunk_offset").as_int());
        opening.chunk_root = parse_hash(chunk_obj.at("chunk_root").as_string());
        if (chunk_obj.as_object().count("chunk_root_path")) {
            const auto& path_node = chunk_obj.at("chunk_root_path");
            if (!path_node.is_array()) {
                throw std::runtime_error("chunk_root_path must be array");
            }
            const auto& path_arr = path_node.as_array();
            opening.chunk_root_path.clear();
            if (!path_arr.empty() && path_arr.front().is_array()) {
                for (const auto& level_node : path_arr) {
                    const auto& level_arr = level_node.as_array();
                    std::vector<Hash> level_hashes;
                    level_hashes.reserve(level_arr.size());
                    for (const auto& elem : level_arr) {
                        level_hashes.push_back(parse_hash(elem.as_string()));
                    }
                    opening.chunk_root_path.push_back(std::move(level_hashes));
                }
            } else {
                std::vector<Hash> level_hashes;
                level_hashes.reserve(path_arr.size());
                for (const auto& elem : path_arr) {
                    level_hashes.push_back(parse_hash(elem.as_string()));
                }
                if (!level_hashes.empty()) {
                    opening.chunk_root_path.push_back(std::move(level_hashes));
                }
            }
        }
        return opening;
    };

    for (const auto& node : openings) {
        RowOpening opening;
        opening.row_index = static_cast<std::uint32_t>(node.at("row_index").as_int());
        RowBackendKind backend = proof.row_commitment.backend;
        if (node.as_object().count("backend")) {
            backend = parse_backend(node.at("backend").as_string());
        }

        if (backend == RowBackendKind::MERKLE) {
            const JsonValue* values_node = nullptr;
            if (node.as_object().count("row_values")) {
                values_node = &node.at("row_values");
            }
            if (values_node) {
                const auto& values = values_node->as_array();
                opening.row_values.reserve(values.size());
                for (const auto& val : values) {
                    opening.row_values.push_back(parse_field(val));
                }
            }
            const JsonValue* proof_node = nullptr;
            if (node.as_object().count("proof")) {
                proof_node = &node.at("proof");
            } else if (node.as_object().count("auth_path")) {
                proof_node = &node.at("auth_path");
            }
            if (!proof_node) {
                throw std::runtime_error("missing Merkle proof data");
            }
            const JsonValue* path_node = proof_node;
            if (proof_node->is_object()) {
                if (!proof_node->as_object().count("path")) {
                    throw std::runtime_error("Merkle proof missing path array");
                }
                path_node = &proof_node->at("path");
            }
            const auto& path_arr = path_node->as_array();
            opening.merkle_path.reserve(path_arr.size());
            for (const auto& elem : path_arr) {
                opening.merkle_path.push_back(parse_hash(elem.as_string()));
            }
            proof.row_openings.push_back(std::move(opening));
        } else if (backend == RowBackendKind::STC) {
            const JsonValue* chunk_obj = nullptr;
            if (node.as_object().count("proof")) {
                chunk_obj = &node.at("proof");
            } else if (node.as_object().count("chunk")) {
                chunk_obj = &node.at("chunk");
            }
            if (!chunk_obj) {
                throw std::runtime_error("missing STC chunk proof");
            }
            RowOpening primary = parse_stc_chunk(*chunk_obj, opening.row_index);
            proof.row_openings.push_back(primary);
            if (node.as_object().count("next_index") && node.as_object().count("next_chunk")) {
                std::uint32_t next_idx = static_cast<std::uint32_t>(node.at("next_index").as_int());
                const auto& next_chunk = node.at("next_chunk");
                RowOpening next_open = parse_stc_chunk(next_chunk, next_idx);
                proof.row_openings.push_back(next_open);
            }
        } else {
            throw std::runtime_error("unsupported backend in row opening");
        }
    }

    if (!proof.row_commitment.row_size && !proof.row_openings.empty()) {
        proof.row_commitment.row_size = static_cast<std::uint32_t>(proof.row_openings.front().row_values.size());
    }

    if (proof.row_commitment.backend == RowBackendKind::MERKLE && proof.row_commitment.row_size == 0) {
        throw std::runtime_error("row size missing for Merkle commitment");
    }

    if (proof.row_commitment.backend == RowBackendKind::STC && proof.row_commitment.chunk_len == 0 && proof.row_commitment.row_size) {
        proof.row_commitment.chunk_len = proof.row_commitment.row_size;
        proof.row_commitment.num_chunks = proof.row_commitment.n_rows;
    }

    return proof;
}

} // namespace geomzk

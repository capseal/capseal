#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "geomzk/geom_air_example.hpp"
#include "geomzk/proof_loader.hpp"
#include "geomzk/verify.hpp"

int main(int argc, char** argv) {
    bool profile = false;
    bool check_air = false;
    bool check_fri = false;
    bool skip_row = false;
    std::string proof_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            profile = true;
        } else if (arg == "--check-air") {
            check_air = true;
        } else if (arg == "--check-fri") {
            check_fri = true;
        } else if (arg == "--skip-row") {
            skip_row = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--profile] [--check-air] [--check-fri] [--skip-row] proof.json" << std::endl;
            return 0;
        } else if (arg.rfind("-", 0) == 0) {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        } else {
            proof_path = arg;
        }
    }
    if (proof_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--profile] [--check-air] [--check-fri] [--skip-row] proof.json" << std::endl;
        return 1;
    }

    try {
        const auto t_parse_start = std::chrono::steady_clock::now();
        geomzk::Proof proof = geomzk::load_proof_from_json_file(proof_path);
        const auto t_parse_end = std::chrono::steady_clock::now();

        geomzk::GeomAirExample air(proof.row_commitment.row_size);
        geomzk::VerifyConfig cfg;
        cfg.check_row_commitments = !skip_row;
        cfg.check_air_constraints = check_air;
        cfg.check_fri = check_fri;

        const auto t_verify_start = std::chrono::steady_clock::now();
        bool ok = geomzk::verify_proof(proof, air, cfg);
        const auto t_verify_end = std::chrono::steady_clock::now();

        if (profile) {
            double parse_sec = std::chrono::duration<double>(t_parse_end - t_parse_start).count();
            double verify_sec = std::chrono::duration<double>(t_verify_end - t_verify_start).count();
            const char* backend_name =
                (proof.row_commitment.backend == geomzk::RowBackendKind::STC) ? "geom_stc_fri" : "geom_plain_fri";
            std::cout << "{\"proof_path\":\"" << proof_path << "\",";
            std::cout << "\"steps\":" << proof.statement.steps << ",";
            std::cout << "\"row_backend\":\"" << backend_name << "\",";
            std::cout << "\"parse_time_sec\":" << parse_sec << ",";
            std::cout << "\"verify_time_sec\":" << verify_sec << ",";
            std::cout << "\"verifier_ok\":" << (ok ? "true" : "false") << "}" << std::endl;
        } else {
            std::cout << (ok ? "VALID" : "INVALID") << std::endl;
        }
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Verification error: " << ex.what() << std::endl;
        return 1;
    }
}

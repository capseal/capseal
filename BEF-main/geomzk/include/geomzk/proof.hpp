#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "geomzk/air.hpp"
#include "geomzk/field.hpp"
#include "geomzk/hash.hpp"

namespace geomzk {

enum class RowBackendKind : std::uint8_t {
    MERKLE = 0,
    STC    = 1,
};

struct RowCommitment {
    RowBackendKind backend = RowBackendKind::MERKLE;
    Hash root{};
    std::uint32_t n_rows = 0;
    std::uint32_t row_size = 0;
    std::uint32_t chunk_len = 0;
    std::uint32_t num_chunks = 0;
    std::uint32_t chunk_tree_arity = 2;
};

struct RowOpening {
    std::uint32_t row_index = 0;
    std::vector<Field> row_values;
    std::vector<Hash> merkle_path;      // for Merkle backend
    std::uint32_t chunk_index = 0;      // for STC backend
    std::uint32_t chunk_offset = 0;
    Hash chunk_root{};
    std::vector<std::vector<Hash>> chunk_root_path;  // path from chunk root to global root
};

struct FriParams {
    std::uint32_t domain_size = 0;
    std::uint32_t max_degree = 0;
    std::uint32_t num_rounds = 0;
    std::uint32_t num_queries = 0;
};

struct FriQuery {
    std::uint32_t base_index = 0;
};

struct FriProof {
    std::vector<Hash> layer_roots;
    std::vector<FriQuery> queries;
};

struct Proof {
    Statement statement;
    FriParams fri_params;
    RowCommitment row_commitment;
    std::vector<RowOpening> row_openings;
    FriProof fri_proof;
};

} // namespace geomzk

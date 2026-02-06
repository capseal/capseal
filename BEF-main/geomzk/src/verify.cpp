#include "geomzk/verify.hpp"

#include <array>
#include <stdexcept>
#include <unordered_map>

#include "geomzk/merkle.hpp"

namespace geomzk {
namespace {

void store_u64_be(std::uint64_t value, std::uint8_t* out) {
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<std::uint8_t>(value & 0xFFu);
        value >>= 8;
    }
}

Hash hash_row_leaf(std::uint32_t row_index, const std::vector<Field>& row) {
    std::vector<std::uint8_t> buf(8 + row.size() * 16);
    store_u64_be(row_index, buf.data());
    for (std::size_t i = 0; i < row.size(); ++i) {
        std::uint64_t value = row[i].raw();
        std::uint8_t* out = buf.data() + 8 + i * 16;
        for (int j = 15; j >= 0; --j) {
            out[j] = static_cast<std::uint8_t>(value & 0xFFu);
            value >>= 8;
        }
    }
    return hash_bytes(buf.data(), buf.size());
}

Hash hash_chunk_leaf(std::uint64_t offset, std::uint32_t local_idx, std::uint64_t value_raw) {
    std::array<std::uint8_t, 8 + 8 + 32> buf{};
    store_u64_be(offset, buf.data());
    store_u64_be(static_cast<std::uint64_t>(local_idx), buf.data() + 8);
    std::uint64_t tmp = value_raw;
    for (int i = 0; i < 32; ++i) {
        buf[8 + 8 + 31 - i] = static_cast<std::uint8_t>(tmp & 0xFFu);
        tmp >>= 8;
    }
    return hash_bytes(buf.data(), buf.size());
}

Hash merkle_from_chunk(const std::vector<Field>& values, std::uint64_t offset) {
    std::vector<Hash> level;
    level.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        level.push_back(hash_chunk_leaf(offset, static_cast<std::uint32_t>(i), values[i].raw()));
    }
    if (level.empty()) {
        throw std::runtime_error("chunk must contain at least one row value");
    }
    std::vector<Hash> next;
    while (level.size() > 1) {
        next.clear();
        for (std::size_t i = 0; i < level.size(); i += 2) {
            const Hash& left = level[i];
            const Hash& right = (i + 1 < level.size()) ? level[i + 1] : level[i];
            next.push_back(hash_concat(left, right));
        }
        level.swap(next);
    }
    return level[0];
}

bool verify_row_opening_merkle(const RowCommitment& commit, const RowOpening& opening) {
    if (opening.row_values.size() != commit.row_size) return false;
    if (opening.row_index >= commit.n_rows) return false;
    Hash leaf = hash_row_leaf(opening.row_index, opening.row_values);
    return merkle_verify(leaf, opening.merkle_path, opening.row_index, commit.root);
}

bool verify_row_opening_stc(const RowCommitment& commit, const RowOpening& opening) {
    if (opening.row_values.size() != commit.row_size) return false;
    if (opening.row_index >= commit.n_rows) return false;
    if (commit.chunk_len == 0 || commit.num_chunks == 0) return false;
    if (opening.chunk_index >= commit.num_chunks) return false;
    const std::uint64_t expected_offset = static_cast<std::uint64_t>(opening.chunk_index) * commit.chunk_len;
    if (opening.chunk_offset != expected_offset) return false;

    Hash chunk_root = merkle_from_chunk(opening.row_values, opening.chunk_offset);
    if (chunk_root != opening.chunk_root) return false;
    const std::uint32_t arity = std::max<std::uint32_t>(2, commit.chunk_tree_arity);
    if (arity == 2) {
        std::vector<Hash> binary_path;
        binary_path.reserve(opening.chunk_root_path.size());
        for (const auto& level : opening.chunk_root_path) {
            if (level.empty()) {
                binary_path.emplace_back();
            } else {
                binary_path.push_back(level.front());
            }
        }
        return merkle_verify(opening.chunk_root, binary_path, opening.chunk_index, commit.root);
    }
    return merkle_verify_kary(
        opening.chunk_root,
        opening.chunk_root_path,
        opening.chunk_index,
        commit.num_chunks,
        arity,
        commit.root
    );
}

bool verify_fri_stub(const FriParams& params, const FriProof& proof) {
    (void)params;
    (void)proof;
    return true;
}

} // namespace

bool verify_proof(const Proof& proof, const Air& air, const VerifyConfig& cfg) {
    const auto& rc = proof.row_commitment;
    if (rc.n_rows != proof.statement.steps) {
        return false;
    }

    std::unordered_map<std::uint32_t, const RowOpening*> opening_map;
    opening_map.reserve(proof.row_openings.size());

    if (cfg.check_row_commitments) {
        for (const auto& opening : proof.row_openings) {
            bool ok = false;
            if (rc.backend == RowBackendKind::MERKLE) {
                ok = verify_row_opening_merkle(rc, opening);
            } else if (rc.backend == RowBackendKind::STC) {
                ok = verify_row_opening_stc(rc, opening);
            }
            if (!ok) {
                return false;
            }
            opening_map[opening.row_index] = &opening;
        }
    } else {
        for (const auto& opening : proof.row_openings) {
            opening_map[opening.row_index] = &opening;
        }
    }

    if (cfg.check_fri) {
        if (!verify_fri_stub(proof.fri_params, proof.fri_proof)) {
            return false;
        }
    }

    if (cfg.check_air_constraints) {
        for (const auto& kv : opening_map) {
            std::uint32_t idx = kv.first;
            const auto* opening = kv.second;
            std::vector<Field> next_row;
            auto next_it = opening_map.find(idx + 1);
            if (next_it != opening_map.end()) {
                next_row = next_it->second->row_values;
            }
            if (!air.check_constraints(idx, opening->row_values, next_row)) {
                return false;
            }
        }
    }

    return true;
}

} // namespace geomzk

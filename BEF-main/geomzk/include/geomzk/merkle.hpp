#pragma once

#include <cstdint>
#include <vector>

#include "geomzk/hash.hpp"

namespace geomzk {

/// Verify a Merkle path from a leaf hash to the expected root.
inline bool merkle_verify(
    const Hash& leaf,
    const std::vector<Hash>& auth_path,
    std::uint32_t leaf_index,
    const Hash& expected_root
) {
    Hash acc = leaf;
    std::uint32_t idx = leaf_index;
    for (const auto& sib : auth_path) {
        if ((idx & 1U) == 0U) {
            acc = hash_concat(acc, sib);
        } else {
            acc = hash_concat(sib, acc);
        }
        idx >>= 1U;
    }
    return acc == expected_root;
}

inline bool merkle_verify_kary(
    const Hash& leaf,
    const std::vector<std::vector<Hash>>& auth_path,
    std::uint32_t leaf_index,
    std::uint32_t total_leaves,
    std::uint32_t arity,
    const Hash& expected_root
) {
    if (arity < 2 || total_leaves == 0) {
        return false;
    }
    Hash acc = leaf;
    std::uint32_t idx = leaf_index;
    std::uint32_t level_size = total_leaves;
    for (const auto& level : auth_path) {
        std::vector<Hash> children;
        children.reserve(arity);
        auto sib_it = level.begin();
        std::uint32_t group_start = (idx / arity) * arity;
        for (std::uint32_t offset = 0; offset < arity; ++offset) {
            std::uint32_t pos = group_start + offset;
            if (pos >= level_size) {
                pos = level_size - 1;
            }
            if (pos == idx) {
                children.push_back(acc);
            } else {
                if (sib_it == level.end()) {
                    return false;
                }
                children.push_back(*sib_it++);
            }
        }
        if (sib_it != level.end()) {
            return false;
        }
        acc = hash_concat_many(children);
        idx /= arity;
        level_size = std::max<std::uint32_t>(1, (level_size + arity - 1) / arity);
    }
    return acc == expected_root;
}

} // namespace geomzk

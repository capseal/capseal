#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace geomzk {

/// 32-byte digest used across the verifier.
struct Hash {
    std::array<std::uint8_t, 32> bytes{};

    bool operator==(const Hash& other) const noexcept { return bytes == other.bytes; }
    bool operator!=(const Hash& other) const noexcept { return !(*this == other); }
};

/// Placeholder hash function. In production you should replace this with
/// BLAKE2s/SHA-256/Poseidon/etc.
Hash hash_bytes(const std::uint8_t* data, std::size_t len);

inline Hash hash_concat(const Hash& left, const Hash& right) {
    std::uint8_t buf[64];
    std::memcpy(buf, left.bytes.data(), 32);
    std::memcpy(buf + 32, right.bytes.data(), 32);
    return hash_bytes(buf, sizeof(buf));
}

Hash hash_concat_many(const std::vector<Hash>& nodes);

} // namespace geomzk

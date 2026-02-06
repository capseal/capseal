#pragma once

#include <cstdint>
#include <stdexcept>

namespace geomzk {

/// Simple prime field wrapper for p = 2^61 - 1.
/// NOTE: replace with a tuned reduction / template if you need different moduli.
class Field {
public:
    static constexpr std::uint64_t MOD = 0x1fffffffffffffffULL; // 2^61 - 1

    Field() : v_(0) {}
    explicit Field(std::uint64_t v) : v_(v % MOD) {}

    static Field zero() { return Field(0); }
    static Field one() { return Field(1); }

    std::uint64_t raw() const { return v_; }

    Field& operator+=(const Field& other) {
        std::uint64_t x = v_ + other.v_;
        if (x >= MOD) x -= MOD;
        v_ = x;
        return *this;
    }

    Field& operator-=(const Field& other) {
        std::uint64_t x = (v_ >= other.v_) ? (v_ - other.v_) : (v_ + MOD - other.v_);
        v_ = x;
        return *this;
    }

    Field& operator*=(const Field& other) {
        __uint128_t prod = static_cast<__uint128_t>(v_) * other.v_;
        std::uint64_t x = static_cast<std::uint64_t>(prod % MOD);
        v_ = x;
        return *this;
    }

    friend Field operator+(Field a, const Field& b) { a += b; return a; }
    friend Field operator-(Field a, const Field& b) { a -= b; return a; }
    friend Field operator*(Field a, const Field& b) { a *= b; return a; }

    bool operator==(const Field& other) const { return v_ == other.v_; }
    bool operator!=(const Field& other) const { return v_ != other.v_; }

private:
    std::uint64_t v_;
};

} // namespace geomzk

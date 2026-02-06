#pragma once

#include "geomzk/air.hpp"

namespace geomzk {

/// Minimal AIR implementation matching the toy JSON example.
/// Each row stores (x1, x2); the transition enforces
///   next.x1 = 2*x1 + x2
///   next.x2 = x1 + x2
class GeomAirExample final : public Air {
public:
    explicit GeomAirExample(std::uint32_t row_size) : row_size_(row_size) {}

    bool check_constraints(
        std::uint32_t row_index,
        const std::vector<Field>& row,
        const std::vector<Field>& next_row
    ) const override {
        (void)row_index;
        if (row.size() != row_size_) {
            return false;
        }
        if (row_size_ < 2) {
            return false;
        }
        if (next_row.empty()) {
            // Last row: no constraints on trailing boundary.
            return true;
        }
        if (next_row.size() != row_size_) {
            return false;
        }
        Field lhs0 = next_row[0];
        Field rhs0 = row[0];
        rhs0 += row[0]; // 2*x1
        rhs0 += row[1];

        Field lhs1 = next_row[1];
        Field rhs1 = row[0];
        rhs1 += row[1];

        return lhs0 == rhs0 && lhs1 == rhs1;
    }

private:
    std::uint32_t row_size_ = 0;
};

} // namespace geomzk

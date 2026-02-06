#pragma once

#include <cstdint>
#include <vector>

#include "geomzk/field.hpp"

namespace geomzk {

struct Statement {
    std::uint32_t steps = 0;
};

class Air {
public:
    virtual ~Air() = default;

    virtual bool check_constraints(
        std::uint32_t row_index,
        const std::vector<Field>& row,
        const std::vector<Field>& next_row
    ) const = 0;
};

} // namespace geomzk

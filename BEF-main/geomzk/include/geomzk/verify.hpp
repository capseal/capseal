#pragma once

#include "geomzk/air.hpp"
#include "geomzk/proof.hpp"

namespace geomzk {

struct VerifyConfig {
    bool check_row_commitments = true;
    bool check_fri = true;
    bool check_air_constraints = true;
};

bool verify_proof(const Proof& proof, const Air& air, const VerifyConfig& cfg = {});

} // namespace geomzk

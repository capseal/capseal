#pragma once

#include <string>

#include "geomzk/proof.hpp"

namespace geomzk {

Proof load_proof_from_json_file(const std::string& path);

} // namespace geomzk

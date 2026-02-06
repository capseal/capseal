#pragma once
#include "enn/types.hpp"
#include <string>
#include <optional>

namespace enn {

struct Calibrator {
    enum class Method {
        Identity,
        Platt
    };

    Method method = Method::Identity;
    F platt_A = 0.0f;
    F platt_B = 0.0f;
    std::string calibrator_id = "identity";

    static Calibrator identity();
    static Calibrator from_json_file(const std::string& path);

    F calibrate(F margin) const;

private:
    static F sigmoid(F x);
};

} // namespace enn

#include "enn/calibrator.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace enn {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibrator file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::optional<std::string> find_string_value(const std::string& content, const std::string& key) {
    auto key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    auto colon = content.find(':', key_pos);
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    auto first_quote = content.find('"', colon + 1);
    if (first_quote == std::string::npos) {
        return std::nullopt;
    }
    auto second_quote = content.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return std::nullopt;
    }
    return content.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::optional<double> find_number_value(const std::string& content, const std::string& key) {
    auto key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    auto colon = content.find(':', key_pos);
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    auto start = content.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) {
        return std::nullopt;
    }
    auto end = content.find_first_not_of("0123456789+-.eE", start);
    auto number_str = content.substr(start, end - start);
    try {
        return std::stod(number_str);
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace

Calibrator Calibrator::identity() {
    Calibrator c;
    c.method = Method::Identity;
    c.calibrator_id = "identity";
    return c;
}

Calibrator Calibrator::from_json_file(const std::string& path) {
    Calibrator calib = Calibrator::identity();
    calib.calibrator_id = path;

    const std::string content = read_file(path);
    auto method_str = find_string_value(content, "method");
    if (method_str && *method_str == "platt") {
        calib.method = Method::Platt;
        calib.platt_A = static_cast<F>(find_number_value(content, "A").value_or(0.0));
        calib.platt_B = static_cast<F>(find_number_value(content, "B").value_or(0.0));
    }

    if (auto id_str = find_string_value(content, "calibrator_id")) {
        calib.calibrator_id = *id_str;
    }

    return calib;
}

F Calibrator::calibrate(F margin) const {
    switch (method) {
        case Method::Identity:
            return sigmoid(margin);
        case Method::Platt: {
            F denom = static_cast<F>(1.0) + std::exp(static_cast<F>(platt_A) * margin + static_cast<F>(platt_B));
            if (denom <= static_cast<F>(1e-9)) {
                return 0.0f;
            }
            return 1.0f / denom;
        }
    }
    return sigmoid(margin);
}

F Calibrator::sigmoid(F x) {
    if (x >= 0) {
        F z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        F z = std::exp(x);
        return z / (1.0f + z);
    }
}

} // namespace enn

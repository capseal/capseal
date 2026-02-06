#include "geomzk/json.hpp"

#include <cctype>
#include <charconv>
#include <cstring>
#include <stdexcept>

namespace geomzk {
namespace detail {

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    JsonValue parse() {
        JsonValue value = parse_value();
        skip_ws();
        if (!eof()) {
            throw std::runtime_error("extra characters after JSON document");
        }
        return value;
    }

private:
    bool eof() const { return pos_ >= text_.size(); }
    char peek() const {
        if (eof()) throw std::runtime_error("unexpected EOF");
        return text_[pos_];
    }
    char get() {
        char c = peek();
        ++pos_;
        return c;
    }
    void skip_ws() {
        while (!eof()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    JsonValue parse_value() {
        skip_ws();
        if (eof()) throw std::runtime_error("unexpected EOF");
        char c = peek();
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error("invalid JSON value");
    }

    JsonValue parse_null() {
        expect_literal("null");
        return JsonValue(JsonValue::Type::Null);
    }

    JsonValue parse_bool() {
        JsonValue v(JsonValue::Type::Bool);
        if (text_.substr(pos_, 4) == "true") {
            pos_ += 4;
            v.bool_value_ = true;
            return v;
        }
        if (text_.substr(pos_, 5) == "false") {
            pos_ += 5;
            v.bool_value_ = false;
            return v;
        }
        throw std::runtime_error("invalid boolean literal");
    }

    JsonValue parse_string() {
        if (get() != '"') throw std::runtime_error("expected string");
        JsonValue v(JsonValue::Type::String);
        while (true) {
            if (eof()) throw std::runtime_error("unterminated string");
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                if (eof()) throw std::runtime_error("bad escape");
                char esc = get();
                switch (esc) {
                    case '"': v.string_value_.push_back('"'); break;
                    case '\\': v.string_value_.push_back('\\'); break;
                    case '/': v.string_value_.push_back('/'); break;
                    case 'b': v.string_value_.push_back('\b'); break;
                    case 'f': v.string_value_.push_back('\f'); break;
                    case 'n': v.string_value_.push_back('\n'); break;
                    case 'r': v.string_value_.push_back('\r'); break;
                    case 't': v.string_value_.push_back('\t'); break;
                    default:
                        throw std::runtime_error("unsupported escape sequence");
                }
            } else {
                v.string_value_.push_back(c);
            }
        }
        return v;
    }

    JsonValue parse_number() {
        std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
            throw std::runtime_error("invalid number");
        }
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
        if (!eof() && peek() == '.') {
            throw std::runtime_error("fractional numbers not supported");
        }
        std::string_view slice = text_.substr(start, pos_ - start);
        long long value = 0;
        auto [ptr, ec] = std::from_chars(slice.data(), slice.data() + slice.size(), value, 10);
        if (ec != std::errc()) {
            throw std::runtime_error("failed to parse integer");
        }
        JsonValue v(JsonValue::Type::Number);
        v.int_value_ = value;
        v.double_value_ = static_cast<double>(value);
        return v;
    }

    JsonValue parse_array() {
        if (get() != '[') throw std::runtime_error("expected array");
        JsonValue v(JsonValue::Type::Array);
        skip_ws();
        if (!eof() && peek() == ']') {
            get();
            return v;
        }
        while (true) {
            v.array_value_.push_back(parse_value());
            skip_ws();
            if (eof()) throw std::runtime_error("unterminated array");
            char c = get();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("expected comma in array");
            skip_ws();
        }
        return v;
    }

    JsonValue parse_object() {
        if (get() != '{') throw std::runtime_error("expected object");
        JsonValue v(JsonValue::Type::Object);
        skip_ws();
        if (!eof() && peek() == '}') {
            get();
            return v;
        }
        while (true) {
            skip_ws();
            JsonValue key = parse_string();
            skip_ws();
            if (get() != ':') throw std::runtime_error("expected colon");
            JsonValue value = parse_value();
            const std::string key_name = key.as_string();
            v.object_value_.emplace(key_name, value);
            skip_ws();
            if (eof()) throw std::runtime_error("unterminated object");
            char c = get();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("expected comma in object");
            skip_ws();
        }
        return v;
    }

    void expect_literal(const char* literal) {
        std::size_t len = std::strlen(literal);
        if (text_.substr(pos_, len) != std::string_view(literal, len)) {
            throw std::runtime_error("unexpected literal");
        }
        pos_ += len;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

} // namespace detail

JsonValue::JsonValue() : type_(Type::Null) {}
JsonValue::JsonValue(Type t) : type_(t) {}

JsonValue JsonValue::parse(std::string_view text) {
    detail::JsonParser parser(text);
    return parser.parse();
}

bool JsonValue::as_bool() const {
    if (!is_bool()) throw std::runtime_error("JSON value is not a bool");
    return bool_value_;
}

std::int64_t JsonValue::as_int() const {
    if (!is_number()) throw std::runtime_error("JSON value is not a number");
    return int_value_;
}

double JsonValue::as_double() const {
    if (!is_number()) throw std::runtime_error("JSON value is not a number");
    return double_value_;
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) throw std::runtime_error("JSON value is not a string");
    return string_value_;
}

const JsonValue::array_t& JsonValue::as_array() const {
    if (!is_array()) throw std::runtime_error("JSON value is not an array");
    return array_value_;
}

const JsonValue::object_t& JsonValue::as_object() const {
    if (!is_object()) throw std::runtime_error("JSON value is not an object");
    return object_value_;
}

const JsonValue& JsonValue::at(const std::string& key) const {
    if (!is_object()) throw std::runtime_error("JSON value is not an object");
    auto it = object_value_.find(key);
    if (it == object_value_.end()) {
        std::string keys;
        for (const auto& kv : object_value_) {
            keys.append(kv.first);
            keys.push_back(',');
        }
        throw std::runtime_error("missing JSON key: " + key + " (available: " + keys + ")");
    }
    return it->second;
}

} // namespace geomzk

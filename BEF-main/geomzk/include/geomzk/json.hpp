#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace geomzk {

namespace detail {
class JsonParser;
}

class JsonValue {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    using array_t = std::vector<JsonValue>;
    using object_t = std::map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(Type t);

    static JsonValue parse(std::string_view text);

    Type type() const { return type_; }

    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const;
    std::int64_t as_int() const;
    double as_double() const;
    const std::string& as_string() const;
    const array_t& as_array() const;
    const object_t& as_object() const;

    const JsonValue& at(const std::string& key) const;
    const JsonValue& operator[](const std::string& key) const { return at(key); }

private:
    friend class detail::JsonParser;
    Type type_;
    bool bool_value_ = false;
    std::int64_t int_value_ = 0;
    double double_value_ = 0.0;
    std::string string_value_;
    array_t array_value_;
    object_t object_value_;
};

} // namespace geomzk

#pragma once
#include <napi.h>
#include <string>
#include <sstream>
#include <iostream>

inline void checkIF(Napi::Env& env, bool error, const std::string& s) {
    if (!error) {
        std::stringstream _error;
        _error << "IF failure\nError: " << s;
        throw Napi::TypeError::New(env, _error.str());
    }
}

template <typename T>
inline void checkLE(Napi::Env& env, T a, T b) {
	checkIF(a >= b, "check le..");
}

template <typename T>
inline void checkGE(Napi::Env& env, T a, T b) {
	checkIF(a < b, "check ge..");
}

template <typename T>
inline void checkEQ(Napi::Env& env, T a, T b) {
	checkIF(a == b, "check eq..");
}

inline bool IsTypeArray(const Napi::Value& a, const napi_typedarray_type& s) {

    switch (s)
    {
    case napi_uint8_array:
        return a.IsTypedArray() && ((a.As<Napi::TypedArray>().TypedArrayType() == napi_uint8_array) || (a.As<Napi::TypedArray>().TypedArrayType() == napi_uint8_clamped_array));
    case napi_int8_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_int8_array;
    case napi_int16_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_int16_array;
    case napi_uint16_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_uint16_array;
    case napi_int32_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_int32_array;
    case napi_uint32_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_uint32_array;
    case napi_float32_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_float32_array;
    case napi_float64_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_float64_array;
    case napi_bigint64_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_bigint64_array;
    case napi_biguint64_array:
        return a.IsTypedArray() && a.As<Napi::TypedArray>().TypedArrayType() == napi_biguint64_array;
    default:
        break;
    }
    return false;
}

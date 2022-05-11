#include "../../include/wrap/MateData.h"

namespace annoa
{
    Napi::FunctionReference MateDataWrap::constructor;
    Napi::Object MateDataWrap::Init(Napi::Env env, Napi::Object exports, std::string keyname) {

        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env,
            keyname.c_str(), {
                InstanceMethod("GetData", &MateDataWrap::GetData),
                InstanceMethod("SetData", &MateDataWrap::SetData),
                InstanceMethod("Normalize", &MateDataWrap::Normalize),
                InstanceMethod("Scale", &MateDataWrap::Scale),
            }
        );
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();

        exports.Set(keyname, func);
        return exports;
    }

    MateDataWrap::MateDataWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<MateDataWrap>(info) {
        Napi::Env env = info.Env();

        int length = info.Length();

        if (length < 3 || !info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()) {
            Napi::TypeError::New(env, "shape error expected").ThrowAsJavaScriptException();
            return;
        }

        int n = 0;
        int c = info[0].ToNumber();
        int h = info[1].ToNumber();
        int w = info[2].ToNumber();
        if (c < 1 || c > 4) {
            Napi::TypeError::New(env, "channel error expected").ThrowAsJavaScriptException();
            return;
        }
        _shape.init(n, c, h, w);
        info.This().ToObject().Set("n", n);
        info.This().ToObject().Set("c", c);
        info.This().ToObject().Set("h", h);
        info.This().ToObject().Set("w", w);
        info.This().ToObject().Set("flag", _flag);
        info.This().ToObject().Set("data", env.Null());
    }

    void MateDataWrap::Destructor(napi_env env, void* nativeObject, void* /*finalize_hint*/) {
        reinterpret_cast<MateDataWrap*>(nativeObject)->~MateDataWrap();
    }

    MateDataWrap::~MateDataWrap() {
        _data = nullptr;
    }
    Napi::Value MateDataWrap::Create(const Napi::Number& channel, const Napi::Number& height, const Napi::Number& width)
    {
        Napi::Object obj = constructor.New({ channel, height, width });

        return obj;
    }

    Napi::Value MateDataWrap::GetData(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Value data = info.This().ToObject().Get("data");

        return data;
    }
    void MateDataWrap::SetData(int flag, Napi::Float32Array& array_) {

        int image_size = _shape.image_size();
        int array_size = array_.ElementLength();
        int lost = array_size % image_size;
        int batch = array_size / image_size;
        if (lost || batch < 1) {
            Napi::TypeError::New(this->Env(), "image size error expected").ThrowAsJavaScriptException();
            return;
        }
        _shape.n = batch;
        _flag = flag;
        Napi::Object self = this->Value();
        self.Set("n", batch);
        self.Set("flag", _flag);
        self.Set("data", array_);
        _data = array_.ArrayBuffer().Data();
    }
    Napi::Value MateDataWrap::SetData(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        int length = info.Length();

        if (length < 1 || !IsTypeArray(info[0], napi_float32_array)) {
            Napi::TypeError::New(env, "you have to set Float32Array expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        Napi::Float32Array array_ = info[0].As< Napi::Float32Array>();
        int image_size = _shape.image_size();
        int array_size = array_.ElementLength();
        int lost = array_size % image_size;
        int batch = array_size / image_size;
        if (lost || batch < 1) {
            Napi::TypeError::New(env, "image size error expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        _shape.n = batch;
        _flag = 0;
        info.This().ToObject().Set("n", batch);
        info.This().ToObject().Set("flag", _flag);
        info.This().ToObject().Set("data", array_);
        _data = array_.ArrayBuffer().Data();
        return env.Null();
    }
    Napi::Value MateDataWrap::Normalize(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
        }

        if (info.Length() < 2)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if ((!info[0].IsArray() || !info[1].IsArray()) &&
            (!IsTypeArray(info[0], napi_float32_array) ||
                !IsTypeArray(info[1], napi_float32_array)))
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        Napi::Float32Array mean_;

        if (info[0].IsArray()) {
            Napi::Array mean_array = info[0].As<Napi::Array>();
            int length_m = mean_array.Length();
            mean_ = Napi::Float32Array::New(env, length_m);
            for (UINT32 i = 0; i < mean_array.Length(); i++) {
                Napi::Value m = mean_array.Get(i);
                if (!m.IsNumber()) {
                    throw Napi::TypeError::New(env, "mean value error....");
                }
                mean_.Set(i, m.ToNumber());
            }
        }
        else if (IsTypeArray(info[0], napi_float32_array)) {

            mean_ = info[0].As<Napi::Float32Array>();
        }
        UINT32 lengthM = mean_.ElementLength();
        float* mean = reinterpret_cast<float*>(mean_.ArrayBuffer().Data());
        Napi::Float32Array std_;
        if (info[1].IsArray()) {

            Napi::Array std_array = info[0].As<Napi::Array>();
            int length_s = std_array.Length();
            std_ = Napi::Float32Array::New(env, length_s);
            for (UINT32 i = 0; i < std_array.Length(); i++) {
                Napi::Value s = std_array.Get(i);
                if (!s.IsNumber()) {
                    throw Napi::TypeError::New(env, "std value error....");
                }
                std_.Set(i, s.ToNumber());
            }
        }
        else if (IsTypeArray(info[1], napi_float32_array)) {

            std_ = info[1].As<Napi::Float32Array>();
        }
        UINT32 lengthS = std_.ElementLength();
        float* stdv = reinterpret_cast<float*>(std_.ArrayBuffer().Data());
        //printf("%d", lengthM);
        if (lengthM != lengthS || lengthM < UINT32(_shape.channel())) {

            throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels < lengthS");
        }
        float scale = 1.0f;
        if (info.Length() == 3 && info[2].IsNumber())
        {
            scale = info[2].ToNumber().FloatValue();
        }

        if (_flag)
        {
            float_to_float_convert_norm_o_cpu(_shape.data_size(), scale, _shape.number(), lengthM, mean, stdv, (const FLOAT*)_data, (FLOAT*)_data);
        }
        else
        {
            float_to_float_convert_norm_cpu(_shape.data_size(), scale, _shape.number(), lengthM, mean, stdv, (const FLOAT*)_data, (FLOAT*)_data);
        }
        return info.This();
    }
    Napi::Value MateDataWrap::Scale(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
        }

        if (info.Length() < 1)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!info[0].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        float scale = info[0].ToNumber().FloatValue();

        scale_norm_cpu(_shape.data_size(), scale, (FLOAT*)_data);
        return info.This();
    }
}

#include "../../include/wrap/Image.h"

namespace annoa
{
    Napi::FunctionReference ImageWrap::constructor;
    Napi::Object ImageWrap::Init(Napi::Env env, Napi::Object exports, std::string keyname) {

        Napi::HandleScope scope(env);
        Napi::Function func = DefineClass(env,
            keyname.c_str(), {
                InstanceMethod("GetNCHWData", &ImageWrap::GetNCHWData),
                InstanceMethod("GetNHWCData", &ImageWrap::GetNHWCData),
                InstanceMethod("SetNCHWData", &ImageWrap::SetNCHWData),
                InstanceMethod("SetNHWCData", &ImageWrap::SetNHWCData),
                InstanceMethod("RemoveAlpha", &ImageWrap::RemoveAlpha),
            }
        );
        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();

        exports.Set(keyname, func);
        return exports;
    }

    ImageWrap::ImageWrap(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<ImageWrap>(info) {
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
        _shape = Shape(n,c,h,w);
        info.This().ToObject().Set("n", n);
        info.This().ToObject().Set("c", c);
        info.This().ToObject().Set("h", h);
        info.This().ToObject().Set("w", w);
        info.This().ToObject().Set("flag", _flag);
        info.This().ToObject().Set("data", env.Null());
    }

    void ImageWrap::Destructor(napi_env env, void* nativeObject, void* /*finalize_hint*/) {
        reinterpret_cast<ImageWrap*>(nativeObject)->~ImageWrap();
    }

    ImageWrap::~ImageWrap() {
        _data = nullptr;
    }
    Napi::Value ImageWrap::GetNCHWData(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            return data;
        }
        if (_flag) {
            _flag = 0;
            Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
            nhwc_to_nchw_cpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
            _data = data_array.ArrayBuffer().Data();
            info.This().ToObject().Set("data", data_array);
            data = data_array;
        }

        return data;
    }
    Napi::Value ImageWrap::GetNHWCData(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            return data;
        }
        if (!_flag) {
            _flag = 1;
            Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
            nchw_to_nhwc_cpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
            _data = data_array.ArrayBuffer().Data();
            info.This().ToObject().Set("data", data_array);
            data = data_array;
        }

        return data;
    }
    Napi::Value ImageWrap::SetNCHWData(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        int length = info.Length();

        if (length < 1 || !IsTypeArray(info[0], napi_uint8_array)) {
            Napi::TypeError::New(env, "you have to set Uint8Array or Uint8ClampedArray expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        Napi::TypedArray array_ = info[0].As< Napi::TypedArray>();
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
    Napi::Value ImageWrap::SetNHWCData(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        int length = info.Length();

        if (length < 1 || !IsTypeArray(info[0], napi_uint8_array)) {
            Napi::TypeError::New(env, "you have to set Uint8Array or Uint8ClampedArray expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        Napi::TypedArray array_ = info[0].As< Napi::TypedArray>();
        int image_size = _shape.image_size();
        int array_size = array_.ElementLength();
        int lost = array_size % image_size;
        int batch = array_size / image_size;
        if (lost || batch < 1) {
            Napi::TypeError::New(env, "image size error expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        _shape.n = batch;
        _flag = 1;
        info.This().ToObject().Set("n", batch);
        info.This().ToObject().Set("flag", _flag);
        info.This().ToObject().Set("data", array_);
        _data = array_.ArrayBuffer().Data();
        return env.Null();
    }
    void ImageWrap::RemoveAlpha(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return;
        }

        if (_shape.channel() != 4) {

            return;
        }
        _shape.c = 3;
        Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
        if (_flag) {
            remove_alpha_cpu(_shape.grid_size(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
        }
        else {
            //remove_alpha_chw_cpu(_shape.data_size(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
            memcpy(data_array.ArrayBuffer().Data(), _data, data_array.ByteLength());
        }
        _data = data_array.ArrayBuffer().Data();
        info.This().ToObject().Set("data", data_array);
    }
}

#include "../../include/wrap/Image.h"
#include "../../include/wrap/MateData.h"

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
                InstanceMethod("HorizontalFlip", &ImageWrap::HorizontalFlip),
                InstanceMethod("RandomCrop", &ImageWrap::RandomCrop),
                InstanceMethod("NormalizeToMateData", &ImageWrap::Normalize),
                InstanceMethod("MateData", &ImageWrap::MateData),
                InstanceMethod("ScaleSize", &ImageWrap::ScaleSize),
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

    Napi::Value ImageWrap::Create(const Napi::Number& number, const Napi::Number& channel, const Napi::Number& height, const Napi::Number& width)
    {
        Napi::Object obj = constructor.New({ number , channel, height, width });

        return obj;
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
            memcpy(_data, data_array.ArrayBuffer().Data(), data_array.ByteLength());// _data = data_array.ArrayBuffer().Data();
            //info.This().ToObject().Set("data", data_array);
            //data = data_array;
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
            memcpy(_data, data_array.ArrayBuffer().Data(), data_array.ByteLength());// _data = data_array.ArrayBuffer().Data();
            //info.This().ToObject().Set("data", data_array);
            //data = data_array;
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
        return info.This();
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
        return info.This();
    }
    Napi::Value ImageWrap::RemoveAlpha(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (_shape.channel() != 4) {

            return env.Null();
        }
        _shape.c = 3;
        Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
        if (_flag) {
            remove_alpha_cpu(_shape.grid_size(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
        }
        else {
            remove_alpha_chw_cpu(_shape.number(), _shape.channel_size() * 4, _shape.image_size(), (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()));
            //memcpy(data_array.ArrayBuffer().Data(), _data, data_array.ByteLength());
        }
        _data = data_array.ArrayBuffer().Data();
        info.This().ToObject().Set("data", data_array);
        return info.This();
    }
    Napi::Value ImageWrap::HorizontalFlip(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (!_flag) {
            horizontal_flip_cpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (UINT8*)(_data));
        }
        else {
            horizontal_flip_nhwc_cpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (UINT8*)(_data));
        }
        return info.This();
    }
    Napi::Value ImageWrap::RandomCrop(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (info.Length() < 3)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }
        if (!info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()) {
            throw Napi::TypeError::New(env, "Wrong type of arguments");
        }
        int h = info[0].ToNumber().Int32Value();
        int w = info[1].ToNumber().Int32Value();
        int p = info[2].ToNumber().Int32Value();
        int ow = _shape.width();
        int oh = _shape.height();
        int channel = _shape.channel();
        if (h > (oh + 2 * p)) {

            throw Napi::TypeError::New(env, "Wrong h check h > oh + 2 * p of arguments");
        }
        if (w > (ow + 2 * p)) {

            throw Napi::TypeError::New(env, "Wrong w check w > ow + 2 * p of arguments");
        }
        _shape.h = h;
        _shape.w = w;
        int size = _shape.data_size();
        Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
        Napi::Int32Array move_array = Napi::Int32Array::New(env, _shape.number() * 2);
        if (!_flag) {
            random_crop_cpu(size, channel, oh, ow, h, w, p, (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()), (int*)(move_array.ArrayBuffer().Data()));
        }
        else {
            random_crop_nhwc_cpu(size, channel, oh, ow, h, w, p, (const UINT8*)(_data), (UINT8*)(data_array.ArrayBuffer().Data()), (int*)(move_array.ArrayBuffer().Data()));
        }
        _data = data_array.ArrayBuffer().Data();
        info.This().ToObject().Set("data", data_array);
        return move_array;
    }
    Napi::Value ImageWrap::Normalize(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
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
        if (lengthM != lengthS || lengthM < (UINT32)_shape.channel()) {

            throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels < lengthS");
        }
        float scale = 1.0f;
        if (info.Length() == 3 && info[2].IsNumber())
        {
            scale = info[2].ToNumber().FloatValue();
        }

        Napi::Float32Array outData = Napi::Float32Array::New(env, _shape.data_size());
        float* result = (float*)outData.ArrayBuffer().Data();
        if (_flag)
        {
            uint8_to_float_convert_norm_o_cpu(_shape.data_size(), scale, _shape.number(), lengthM, mean, stdv, (UINT8*)_data, result);
        }
        else
        {
            uint8_to_float_convert_norm_cpu(_shape.data_size(), scale, _shape.number(), lengthM, mean, stdv, (UINT8*)_data, result);
        }
        //Napi::Number number = Napi::Number::New(env, _shape.number());
        Napi::Number channel = Napi::Number::New(env, _shape.channel());
        Napi::Number height = Napi::Number::New(env, _shape.height());
        Napi::Number width = Napi::Number::New(env, _shape.width());

        Napi::Value mate = MateDataWrap::Create(channel, height, width);
        Napi::Object mate_o = mate.ToObject();
        MateDataWrap *mate_ = Napi::ObjectWrap<MateDataWrap>::Unwrap(mate_o);
        mate_->SetData(_shape, outData);
        return mate;
    }
    Napi::Value ImageWrap::MateData(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        float scale = 1.0f;
        if (info.Length() == 1 && info[0].IsNumber())
        {
            scale = info[0].ToNumber().FloatValue();
        }

        Napi::Float32Array outData = Napi::Float32Array::New(env, _shape.data_size());
        float* result = (float*)outData.ArrayBuffer().Data();
        scale_norm_cpu(_shape.data_size(), scale, (UINT8*)_data, result);
        //Napi::Number number = Napi::Number::New(env, _shape.number());
        Napi::Number channel = Napi::Number::New(env, _shape.channel());
        Napi::Number height = Napi::Number::New(env, _shape.height());
        Napi::Number width = Napi::Number::New(env, _shape.width());

        Napi::Value mate = MateDataWrap::Create(channel, height, width);
        Napi::Object mate_o = mate.ToObject();
        MateDataWrap *mate_ = Napi::ObjectWrap<MateDataWrap>::Unwrap(mate_o);
        mate_->SetData(_shape, outData);
        return mate;
    }
    Napi::Value ImageWrap::ScaleSize(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        if (info.Length() < 2)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!info[0].IsNumber() ||
            !info[1].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        int scaleh = info[0].ToNumber().Int32Value();
        int scalew = info[1].ToNumber().Int32Value();

        int oh = _shape.height();
        int ow = _shape.width();
        float sh = static_cast<float>(oh * scaleh);
        float sw = static_cast<float>(ow * scalew);

        if (sh == oh && ow == sw)
        {
            return info.This();
        }
        UINT32 length = _shape.data_size();
        UINT32 channels = _shape.channel();
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, _shape.number() * channels * scaleh * scalew);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        uint8_to_uint8_scale_cpu(length, img_data, _shape, sh, sw, result, !_flag);

        _shape.h = scaleh;
        _shape.w = scalew;
        _data = result;
        info.This().ToObject().Set("data", outData);
        return info.This();
    }
}

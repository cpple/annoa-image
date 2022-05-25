
#include <random>
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
                InstanceMethod("ColorHSV", &ImageWrap::ColorHSV),
                InstanceMethod("CaptureImgByBoundingBox", &ImageWrap::CaptureImgByBoundingBox),
                InstanceMethod("GreyScale", &ImageWrap::GreyScale),

                InstanceMethod("ScaleSizeGPU", &ImageWrap::ScaleSizeGPU),
                InstanceMethod("RandomCropGPU", &ImageWrap::RandomCropGPU),
                InstanceMethod("HorizontalFlipGPU", &ImageWrap::HorizontalFlipGPU),
                InstanceMethod("ColorHSVGPU", &ImageWrap::ColorHSVGPU),
                InstanceMethod("NormalizeToMateDataGPU", &ImageWrap::NormalizeGPU),
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
        _has_alpha = false;
        int c = info[0].ToNumber();
        int h = info[1].ToNumber();
        int w = info[2].ToNumber();
        if (length == 4 && info[3].IsBoolean()) {
            _has_alpha = info[3].ToBoolean();
        }
        if (c < 1 || c > 4) {
            Napi::TypeError::New(env, "channel error expected").ThrowAsJavaScriptException();
            return;
        }
        if (c == 4) {
            _has_alpha = true;
        }
        _shape.init(n,c,h,w);
        info.This().ToObject().Set("n", n);
        info.This().ToObject().Set("c", c);
        info.This().ToObject().Set("h", h);
        info.This().ToObject().Set("w", w);
        info.This().ToObject().Set("has_alpha", _has_alpha);
        info.This().ToObject().Set("flag", _flag);
        info.This().ToObject().Set("data", env.Null());
    }

    void ImageWrap::Destructor(napi_env env, void* nativeObject, void* /*finalize_hint*/) {
        reinterpret_cast<ImageWrap*>(nativeObject)->~ImageWrap();
    }

    ImageWrap::~ImageWrap() {
        _data = nullptr;
    }

    Napi::Value ImageWrap::Create(const Napi::Number& channel, const Napi::Number& height, const Napi::Number& width)
    {
        Napi::Object obj = constructor.New({ channel, height, width });

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

        if (_shape.channel() != 4 && !_has_alpha) {

            return info.This();
        }
        if (!_has_alpha) {
            return info.This();
        }
        const int c = _shape.c - 1;
        _has_alpha = false;
        int length = _shape.data_size();
        UINT32 pix_count = _shape.grid_size();
        Napi::Uint8Array out = Napi::Uint8Array::New(env, (pix_count * c));
        UINT8 * result = reinterpret_cast<UINT8 *>(out.ArrayBuffer().Data());
        if (_flag) {
            remove_alpha_cpu(pix_count, c, (const UINT8*)(_data), result);
        }
        else {
            remove_alpha_chw_cpu(_shape.number(), _shape.image_size(), c * _shape.channel_size(), (const UINT8*)(_data), result);
            //memcpy(data_array.ArrayBuffer().Data(), _data, data_array.ByteLength());
        }
        _shape.c = c;
        _data = result;
        info.This().ToObject().Set("has_alpha", _has_alpha);
        info.This().ToObject().Set("c", c);
        info.This().ToObject().Set("data", out);
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
        info.This().ToObject().Set("h", _shape.h);
        info.This().ToObject().Set("w", _shape.w);
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
        //Napi::Number channel = Napi::Number::New(env, _shape.channel());
        //Napi::Number height = Napi::Number::New(env, _shape.height());
        //Napi::Number width = Napi::Number::New(env, _shape.width());

        //Napi::Value mate = MateDataWrap::Create(channel, height, width);
        //Napi::Object mate_o = mate.ToObject();
        //MateDataWrap *mate_ = Napi::ObjectWrap<MateDataWrap>::Unwrap(mate_o);
        //mate_->SetData(_flag, outData);
        return outData;
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
        //Napi::Number channel = Napi::Number::New(env, _shape.channel());
        //Napi::Number height = Napi::Number::New(env, _shape.height());
        //Napi::Number width = Napi::Number::New(env, _shape.width());

        //Napi::Value mate = MateDataWrap::Create(channel, height, width);
        //Napi::Object mate_o = mate.ToObject();
        //MateDataWrap *mate_ = Napi::ObjectWrap<MateDataWrap>::Unwrap(mate_o);
        //mate_->SetData(_flag, outData);
        return outData;
    }
    Napi::Value ImageWrap::ScaleSize(const Napi::CallbackInfo& info) {

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

        if (!info[0].IsNumber() ||
            !info[1].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        int scaleh = info[0].ToNumber().Int32Value();
        int scalew = info[1].ToNumber().Int32Value();

        int oh = _shape.height();
        int ow = _shape.width();
        //float sh = static_cast<float>(oh * scaleh);
        //float sw = static_cast<float>(ow * scalew);

        if (scaleh == oh && ow == scalew)
        {
            return info.This();
        }
        UINT32 channels = _shape.channel();
        UINT32 length = _shape.number() * scaleh * scalew;
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, length * channels);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        //throw Napi::TypeError::New(env, "Wrong arguments" + std::to_string(!_flag));
        uint8_to_uint8_scale_cpu(length, img_data, _shape, scaleh, scalew, result, _flag);
        _shape.h = scaleh;
        _shape.w = scalew;

        _data = result;
        info.This().ToObject().Set("data", outData);
        info.This().ToObject().Set("h", _shape.h);
        info.This().ToObject().Set("w", _shape.w);
        return info.This();
    }

    Napi::Value ImageWrap::ColorHSV(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        Napi::Value data = args.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (args.Length() < 3)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!args[0].IsNumber() || !args[1].IsNumber() ||
            !args[2].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        float hue = args[0].ToNumber().FloatValue();
        float sat = args[1].ToNumber().FloatValue();
        float val = args[2].ToNumber().FloatValue();

        UINT32 length = _shape.data_size();
        UINT32 pixels = _shape.grid_size();
        UINT32 channels = _shape.channel();
        if (channels < 3 || channels > 4)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, length);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        uint8_to_uint8_color_cpu(pixels, img_data, _shape, hue, sat, val, result, _flag);
        _data = result;
        args.This().ToObject().Set("data", outData);

        return args.This();
    }
    Napi::Value ImageWrap::CaptureImgByBoundingBox(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        if (args.Length() < 1)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!args[0].IsArray())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }
        int length = _shape.data_size();
        int batch = _shape.number();
        int channels = _shape.channel();
        int height = _shape.height();
        int width = _shape.width();

        Napi::Array bboxList = args[0].As<Napi::Array>();
        int batch_idx = 0;
        if (args.Length() == 2 && args[1].IsNumber()) {
            batch_idx = args[1].ToNumber().Int32Value();
            if (batch_idx < 0 || batch_idx >= batch) {

                throw Napi::TypeError::New(env, "Wrong batch idx arguments");
            }
        }
        int offset = batch_idx * _shape.image_size();
        UINT8* img_data = static_cast<UINT8*>(_data);
        UINT8* TEST_1 = img_data + offset;
        //UINT8* TEST_2 = &img_data[offset];
        //printf("TEST_1:%p TEST_2:%p", TEST_1, TEST_2);
        Napi::Array imgList = Napi::Array::New(env);
        UINT32 size = bboxList.Length();
        for (UINT32 c = 0; c < size; c++) {

            Napi::Value bboxV = bboxList.Get(c);
            if (!IsTypeArray(bboxV, napi_uint32_array) && !IsTypeArray(bboxV, napi_float32_array))
            {
                throw Napi::TypeError::New(env, "bbox data type error arguments");
            }
            bool isFloat = false;
            if (IsTypeArray(bboxV, napi_float32_array))
            {
                isFloat = true;
            }
            Napi::TypedArray bbox = bboxList.Get(c).As<Napi::TypedArray>();
            if (bbox.ElementLength() != 4)
            {
                throw Napi::TypeError::New(env, "bbox length error arguments");
            }
            void* data = bbox.ArrayBuffer().Data();

            float* dataF = nullptr;
            UINT32* dataU = nullptr;
            if (isFloat)
                dataF = static_cast<float*>(data);
            else
                dataU = static_cast<UINT32*>(data);

            int Bx = static_cast<int>(isFloat ? dataF[0] : dataU[0]);
            int By = static_cast<int>(isFloat ? dataF[1] : dataU[1]);
            int Bw = static_cast<int>(isFloat ? dataF[2] : dataU[2]);
            int Bh = static_cast<int>(isFloat ? dataF[3] : dataU[3]);
            int hw = Bw / 2;
            int hh = Bh / 2;

            int x1 = Bx - hw;
            int y1 = By - hh;
            int x2 = x1 + Bw;// -hw;
            int y2 = y1 + Bh;// -hh;

            if (x1 < 0) {
                x1 = 0;
            }
            if (y1 < 0) {
                y1 = 0;
            }
            if (x1 > width) {
                continue;
            }
            if (y1 > height) {
                continue;
            }

            if (x2 > width) {
                x2 = width;
            }
            if (y2 > height) {
                y2 = height;
            }

            if (x2 <= x1) {
                continue;
            }
            if (y2 <= y1) {
                continue;
            }

            Bw = x2 - x1;
            Bh = y2 - y1;

            if (Bw < 0 || Bh < 0)
            {
                continue;
            }
            //printf("%d %d %d %d  %d %d  %d", x1, y1, x2, y2, Bw, Bh, channels);

            int spDim = Bw * Bh * channels;
            int picDim = Bw * Bh;

            Napi::Uint8Array outData = Napi::Uint8Array::New(env, spDim);
            UINT8* result = reinterpret_cast<UINT8*>(outData.ArrayBuffer().Data());

            capture_bbox_img_cpu(picDim, TEST_1, channels, width, height, x1, y1, Bw, Bh, result, _flag);

            Napi::Value image = ImageWrap::Create(Napi::Number::New(env, channels), Napi::Number::New(env, Bh), Napi::Number::New(env, Bw));
            Napi::Object image_o = image.ToObject();
            ImageWrap* image_ = Napi::ObjectWrap<ImageWrap>::Unwrap(image_o);
            image_->SetData(_flag, outData);

            imgList.Set(imgList.Length(), image);
        }
        return imgList;
    }
    void ImageWrap::SetData(int flag, Napi::Uint8Array& array_) {

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

    Napi::Value ImageWrap::GreyScale(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        Napi::Value data = args.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (_shape.c < 3) {

            Napi::TypeError::New(env, "rgb data are not intact expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        bool remove_alpha = false;
        bool rgb_merge = false;
        float gamma = 2.2f;
        if (args.Length() > 0 && args[0].IsBoolean())
        {
            remove_alpha = args[0].ToBoolean();
        }
        if (args.Length() > 1 && args[1].IsBoolean())
        {
            rgb_merge = args[1].ToBoolean();
        }

        if (args.Length() > 2 && !args[2].IsNumber())
        {
            gamma = args[2].ToNumber().FloatValue();
        }
        int channel = _shape.c;
        bool has_alpha_old = _has_alpha;
        if (!has_alpha_old) {
            remove_alpha = true;
        }
        if (remove_alpha && _has_alpha) {
            channel -= 1;
            _has_alpha = false;
        }
        if (rgb_merge) {
            channel = _has_alpha ? 2 : 1;
        }

        UINT32 length = _shape.data_size();
        UINT32 pixels = _shape.grid_size();
        UINT32 channels = _shape.channel();
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, channel * pixels);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();
        uint8_to_uint8_grey_cpu(pixels, img_data, _shape, channel, has_alpha_old, gamma, result, _flag);
        _shape.c = channel;
        _data = result;
        args.This().ToObject().Set("has_alpha", _has_alpha);
        args.This().ToObject().Set("c", channel);
        args.This().ToObject().Set("data", outData);

        return args.This();
    }
    Napi::Value ImageWrap::ScaleSizeGPU(const Napi::CallbackInfo& info) {

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

        if (!info[0].IsNumber() ||
            !info[1].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        int scaleh = info[0].ToNumber().Int32Value();
        int scalew = info[1].ToNumber().Int32Value();

        int oh = _shape.height();
        int ow = _shape.width();
        //float sh = static_cast<float>(oh * scaleh);
        //float sw = static_cast<float>(ow * scalew);

        if (scaleh == oh && ow == scalew)
        {
            return info.This();
        }
        UINT32 channels = _shape.channel();
        UINT32 length = _shape.number() * scaleh * scalew;
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, length * channels);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        UINT32 DIM_SIZE_N = outData.ArrayBuffer().ByteLength();
        UINT32 DIM_SIZE_O = _shape.data_size() * sizeof(UINT8);
        //throw Napi::TypeError::New(env, "Wrong arguments" + std::to_string(!_flag));
        void* source_gpu = AnnoaCuda::AnnoaMallocCopyDevice(DIM_SIZE_O, img_data);
        void* new_gpu = AnnoaCuda::AnnoaMallocCopyDevice(DIM_SIZE_N, result);
        uint8_to_uint8_scale_gpu(length, (const UINT8*)source_gpu, _shape, scaleh, scalew, (UINT8*)new_gpu, _flag);
        AnnoaCuda::AnnoaDeviceCopyHost(new_gpu, result, DIM_SIZE_N);
        checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
        AnnoaCuda::AnnoaFreeMemDevice(source_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(new_gpu);
        source_gpu = nullptr;
        new_gpu = nullptr;
        checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
        _shape.h = scaleh;
        _shape.w = scalew;

        _data = result;
        info.This().ToObject().Set("data", outData);
        info.This().ToObject().Set("h", _shape.h);
        info.This().ToObject().Set("w", _shape.w);
        return info.This();
    }

    Napi::Value ImageWrap::RandomCropGPU(const Napi::CallbackInfo& info) {

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
        int size_o = _shape.data_size();
        _shape.h = h;
        _shape.w = w;
        int size = _shape.data_size();
        int number = _shape.number();
        Napi::Uint8Array data_array = Napi::Uint8Array::New(env, _shape.data_size());
        Napi::Int32Array move_array = Napi::Int32Array::New(env, number * 2);

        std::vector<INT32> random_h(number);
        std::vector<INT32> random_w(number);
        std::random_device rd;
        int* move_ptr = static_cast<int*>(move_array.ArrayBuffer().Data());
        if (p > 0)
        {
            for (int ndx = 0; ndx < number; ndx++) {

                random_h[ndx] = (rd() % ((oh + 2 * p) - h)) - p;
                random_w[ndx] = (rd() % ((ow + 2 * p) - w)) - p;
                move_ptr[ndx * 2] = random_h[ndx];
                move_ptr[ndx * 2 + 1] = random_w[ndx];
            }
        }

        void* source_gpu = AnnoaCuda::AnnoaMallocCopyDevice(size_o * sizeof(UINT8), _data);
        void* new_gpu = AnnoaCuda::AnnoaMallocCopyDevice(data_array.ByteLength());
        void* new_move_gpu = nullptr;// AnnoaCuda::AnnoaMallocCopyDevice(move_array.ByteLength());

        void* random_h_gpu = AnnoaCuda::AnnoaMallocCopyDevice(number * sizeof(UINT32), random_h.data());
        void* random_w_gpu = AnnoaCuda::AnnoaMallocCopyDevice(number * sizeof(UINT32), random_w.data());

        //printf("random_crop_gpu random_crop_gpu:%d %d %d %d %d", oh, ow, h, w, p);
        if (!_flag) {
            random_crop_gpu(size, channel, oh, ow, h, w, p, (const INT32*)random_h_gpu, (const INT32*)random_w_gpu, (const UINT8*)(source_gpu), (UINT8*)(new_gpu), (int*)(new_move_gpu));
        }
        else {
            random_crop_nhwc_gpu(size, channel, oh, ow, h, w, p, (const INT32*)random_h_gpu, (const INT32*)random_w_gpu, (const UINT8*)(source_gpu), (UINT8*)(new_gpu), (int*)(new_move_gpu));
        }
        checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
        AnnoaCuda::AnnoaDeviceCopyHost(new_gpu, data_array.ArrayBuffer().Data(), data_array.ByteLength());
        //AnnoaCuda::AnnoaDeviceCopyHost(new_move_gpu, move_array.ArrayBuffer().Data(), move_array.ByteLength());
        AnnoaCuda::AnnoaFreeMemDevice(source_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(new_gpu);
        //AnnoaCuda::AnnoaFreeMemDevice(new_move_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(random_h_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(random_w_gpu);
        random_h.clear();
        random_w.clear();
        source_gpu = nullptr;
        new_gpu = nullptr;
        new_move_gpu = nullptr;
        random_h_gpu = nullptr;
        random_w_gpu = nullptr;
        _data = data_array.ArrayBuffer().Data();
        info.This().ToObject().Set("data", data_array);
        info.This().ToObject().Set("h", _shape.h);
        info.This().ToObject().Set("w", _shape.w);
        return move_array;
    }

    Napi::Value ImageWrap::HorizontalFlipGPU(const Napi::CallbackInfo& info) {

        Napi::Env env = info.Env();

        Napi::Value data = info.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }
        void* source_gpu = AnnoaCuda::AnnoaMallocCopyDevice(_shape.data_size() * sizeof(UINT8), _data);

        if (!_flag) {
            horizontal_flip_gpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (UINT8*)(source_gpu));
        }
        else {
            horizontal_flip_nhwc_gpu(_shape.grid_size(), _shape.channel(), _shape.height(), _shape.width(), (UINT8*)(source_gpu));
        }
        AnnoaCuda::AnnoaDeviceCopyHost(source_gpu, _data, _shape.data_size() * sizeof(UINT8));
        AnnoaCuda::AnnoaFreeMemDevice(source_gpu);
        source_gpu = nullptr;
        return info.This();
    }

    Napi::Value ImageWrap::ColorHSVGPU(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        Napi::Value data = args.This().ToObject().Get("data");

        if (data.IsNull()) {

            Napi::TypeError::New(env, "have no image data expected").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (args.Length() < 3)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!args[0].IsNumber() || !args[1].IsNumber() ||
            !args[2].IsNumber())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        float hue = args[0].ToNumber().FloatValue();
        float sat = args[1].ToNumber().FloatValue();
        float val = args[2].ToNumber().FloatValue();

        UINT32 length = _shape.data_size();
        UINT32 pixels = _shape.grid_size();
        UINT32 channels = _shape.channel();
        if (channels < 3 || channels > 4)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }
        UINT8* img_data = reinterpret_cast<UINT8*>(_data);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, length);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        void* source_gpu = AnnoaCuda::AnnoaMallocCopyDevice(outData.ByteLength(), img_data);
        void* new_gpu = AnnoaCuda::AnnoaMallocCopyDevice(outData.ByteLength());
        uint8_to_uint8_color_gpu(pixels, (UINT8*)source_gpu, _shape, hue, sat, val, (UINT8*)new_gpu, _flag);
        AnnoaCuda::AnnoaDeviceCopyHost(new_gpu, result, outData.ByteLength());
        AnnoaCuda::AnnoaFreeMemDevice(source_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(new_gpu);
        source_gpu = nullptr;
        new_gpu = nullptr;
        _data = result;
        args.This().ToObject().Set("data", outData);

        return args.This();
    }
    Napi::Value ImageWrap::NormalizeGPU(const Napi::CallbackInfo& info) {

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
        void* source_gpu = AnnoaCuda::AnnoaMallocCopyDevice(_shape.data_size() * sizeof(UINT8), _data);
        void* new_gpu = AnnoaCuda::AnnoaMallocCopyDevice(outData.ByteLength());
        void* mean_gpu = AnnoaCuda::AnnoaMallocCopyDevice(mean_.ByteLength(), mean);
        void* stdv_gpu = AnnoaCuda::AnnoaMallocCopyDevice(std_.ByteLength(), stdv);
        if (_flag)
        {
            uint8_to_float_convert_norm_o_gpu(_shape.data_size(), scale, _shape.number(), lengthM, (const float*)mean_gpu, (const float*)stdv_gpu, (UINT8*)source_gpu, (float*)new_gpu);
        }
        else
        {
            uint8_to_float_convert_norm_gpu(_shape.data_size(), scale, _shape.number(), lengthM, (const float*)mean_gpu, (const float*)stdv_gpu, (UINT8*)source_gpu, (float*)new_gpu);
        }
        AnnoaCuda::AnnoaDeviceCopyHost(new_gpu, result, outData.ByteLength());
        AnnoaCuda::AnnoaFreeMemDevice(source_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(new_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(mean_gpu);
        AnnoaCuda::AnnoaFreeMemDevice(stdv_gpu);
        source_gpu = nullptr;
        new_gpu = nullptr;
        mean_gpu = nullptr;
        stdv_gpu = nullptr;
        //Napi::Number number = Napi::Number::New(env, _shape.number());
        //Napi::Number channel = Napi::Number::New(env, _shape.channel());
        //Napi::Number height = Napi::Number::New(env, _shape.height());
        //Napi::Number width = Napi::Number::New(env, _shape.width());

        //Napi::Value mate = MateDataWrap::Create(channel, height, width);
        //Napi::Object mate_o = mate.ToObject();
        //MateDataWrap *mate_ = Napi::ObjectWrap<MateDataWrap>::Unwrap(mate_o);
        //mate_->SetData(_flag, outData);
        return outData;
    }
}

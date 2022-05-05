#include <napi.h>
#include <cmath>
#include <random>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include <string>
#include "../include/util.h"
#include "../include/tool.h"
#include "../include/wrap/Image.h"
#include "../include/wrap/MateData.h"
#define NODE_LESS_THAN (!(NODE_VERSION_AT_LEAST(0, 5, 4)))
namespace annoa
{
    Napi::Value removeImgBufferAlpha(const Napi::CallbackInfo& info)
	{
        Napi::Env env = info.Env();

		if (info.Length() < 1)
		{
            throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!IsTypeArray(info[0], napi_uint8_array))
		{
            throw Napi::TypeError::New(env, "Wrong arguments");
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

        Napi::Uint8Array imgU8 = info[0].As<Napi::Uint8Array>();
        length = imgU8.ElementLength();
        img_data = reinterpret_cast<UINT8 *>(imgU8.ArrayBuffer().Data());

		if (length % 4 != 0 || !length)
		{
            throw Napi::TypeError::New(env, "img channels error arguments");
		}

		UINT32 bytes = (length / 4) * 3 * sizeof(UINT8);
		UINT32 pix_count = (length / 4);
        Napi::Uint8Array out = Napi::Uint8Array::New(env, bytes / sizeof(UINT8));
		UINT8 * result = reinterpret_cast<UINT8 *>(out.ArrayBuffer().Data());

		remove_alpha_cpu(pix_count, img_data, result);

        return out;
	}

    Napi::Value captureImgByBoundingBox(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 4)
		{
            throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!IsTypeArray(args[0], napi_uint8_array) || !args[1].IsNumber() || !args[2].IsNumber() || !args[3].IsNumber() || (!args[4].IsArray() && !IsTypeArray(args[4], napi_uint32_array) && !IsTypeArray(args[4], napi_float32_array)))
		{
            throw Napi::TypeError::New(env, "Wrong arguments");
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();
        length = imgU8.ElementLength();
        img_data = reinterpret_cast<UINT8 *>(imgU8.ArrayBuffer().Data());

		UINT32 channels = args[1].ToNumber().Uint32Value();
		UINT32 height = args[2].ToNumber().Uint32Value();
		UINT32 width = args[3].ToNumber().Uint32Value();

		if (length % channels != 0 || length != channels * height * width)
		{
			throw Napi::TypeError::New(env, "img channels error arguments");
		}

        Napi::Array bboxList = Napi::Array::New(env);
        if (args[4].IsTypedArray())
        {
            Napi::TypedArray bbox = args[4].As<Napi::TypedArray>();
            bboxList.Set(bboxList.Length(), bbox);
        } else {
			bboxList = args[4].As<Napi::Array>();
		}
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

			UINT32 Bx = static_cast<UINT32>(isFloat ? dataF[0] : dataU[0]);
			UINT32 By = static_cast<UINT32>(isFloat ? dataF[1] : dataU[1]);
			UINT32 Bw = static_cast<UINT32>(isFloat ? dataF[2] : dataU[2]);
			UINT32 Bh = static_cast<UINT32>(isFloat ? dataF[3] : dataU[3]);
			UINT32 hw = Bw / 2;
			UINT32 hh = Bh / 2;

			UINT32 x1 = Bx - hw;
			UINT32 y1 = By - hh;
			UINT32 x2 = x1 + Bw;// -hw;
			UINT32 y2 = y1 + Bh;// -hh;

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

			UINT32 spDim = Bw * Bh * channels;
			UINT32 picDim = Bw * Bh;

            Napi::Uint8Array outData = Napi::Uint8Array::New(env, spDim);
			UINT8 * result = reinterpret_cast<UINT8*>(outData.ArrayBuffer().Data());

			capture_bbox_img_cpu(picDim, img_data, channels, width, height, x1, y1, Bw, Bh, result, true);
            Napi::Object obj = Napi::Object::New(env);
            obj.Set("width", static_cast<double>(Bw));
            obj.Set("height", static_cast<double>(Bh));
            obj.Set("channels", static_cast<double>(channels));
            obj.Set("data", outData);

            imgList.Set(imgList.Length(), obj);
		}
		return imgList;
	}

	Napi::Value convertImgDataToNetData(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 2)
		{
		    throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!IsTypeArray(args[0], napi_uint8_array))
		{
			throw Napi::TypeError::New(env, "Wrong arguments");
		}

		float scale = 1.0f;
		if (args[1].IsNumber())
		{
			scale = args[1].ToNumber().FloatValue();
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();
        length = imgU8.ElementLength();
        img_data = reinterpret_cast<UINT8*>(imgU8.ArrayBuffer().Data());

		UINT32 bytes = length * sizeof(float);
        Napi::Float32Array out = Napi::Float32Array::New(env, length);
		float * result = reinterpret_cast<float*>(out.ArrayBuffer().Data());

		uint8_to_float_convert_cpu(length, scale, img_data, result);

		return out;
	}

	Napi::Value imgRandomCropHorizontalFlipNormalize(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 13)
		{
            throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!IsTypeArray(args[0], napi_uint8_array) || !args[1].IsNumber() || !args[2].IsNumber() || !args[3].IsNumber() ||
			!args[4].IsNumber() || !args[5].IsNumber() || !args[6].IsNumber() || !args[7].IsNumber() || !args[8].IsBoolean() || !args[9].IsBoolean() || !args[10].IsBoolean() ||
			(!(args[11].IsNull() || args[11].IsUndefined()) && !IsTypeArray(args[11], napi_float32_array)) ||
			(!(args[11].IsNull() || args[11].IsUndefined()) && !IsTypeArray(args[11], napi_float32_array)))
		{
			throw Napi::TypeError::New(env, "Wrong arguments");
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;
		UINT32 lengthM = 0;
		UINT32 lengthS = 0;
		UINT32 lengthNew = 0;
		UINT32 channels = 0;

        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();
        length = imgU8.ElementLength();
        img_data = reinterpret_cast<UINT8*>(imgU8.ArrayBuffer().Data());

		float scale = args[1].ToNumber().FloatValue();
		UINT32 batch = args[2].ToNumber().Uint32Value();
		UINT32 oh = args[3].ToNumber().Uint32Value();
		UINT32 ow = args[4].ToNumber().Uint32Value();
		UINT32 h = args[5].ToNumber().Uint32Value();
		UINT32 w = args[6].ToNumber().Uint32Value();
		UINT32 p = args[7].ToNumber().Uint32Value();
		bool channelFirst = args[8].ToBoolean().Value();
		bool hori = args[9].ToBoolean().Value();
		bool norm = args[10].ToBoolean().Value();
		float* mean = nullptr;
		float* stdv = nullptr;
		std::random_device rd;

		if (hori) {
			hori = rd() & 1;
		}

		channels = length / (oh * ow * batch);
		lengthNew = h * w * channels * batch;
		if (norm && IsTypeArray(args[11], napi_float32_array) && IsTypeArray(args[12], napi_float32_array)) {

            Napi::Float32Array mean_ = args[11].As<Napi::Float32Array>();
			lengthM = mean_.ElementLength();
			mean = reinterpret_cast<float*>(mean_.ArrayBuffer().Data());
            Napi::Float32Array std_ = args[12].As<Napi::Float32Array>();
			lengthS = std_.ElementLength();
			stdv = reinterpret_cast<float*>(std_.ArrayBuffer().Data());
			if (channels != lengthM || channels != lengthS) {

				throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels != lengthS");
			}
		}

		if (oh <= p || ow <= p) {

			throw Napi::TypeError::New(env, "Wrong arguments oh + p < h");
		}

		if (oh + p < h) {

			throw Napi::TypeError::New(env, "Wrong arguments oh + p < h");
		}

		if (ow + p < w) {

			throw Napi::TypeError::New(env, "Wrong arguments ow + p < w");
		}

		if (channels < 1 && channels > 4) {

			throw Napi::TypeError::New(env, "Wrong arguments channels < 1 && channels > 4");
		}
        UINT32 sh = 0;
        UINT32 sw = 0;
        if (p > 0)
        {
            sh = rd() % ((oh + 2 * p) - h);
            sw = rd() % ((ow + 2 * p) - w);
        }

        Napi::Float32Array outData = Napi::Float32Array::New(env, lengthNew);
		float* result = (float*)outData.ArrayBuffer().Data();

		if (channelFirst)
		{
			uint8_to_float_random_crop_hori_norm_o_cpu(lengthNew, scale, batch, channels, oh, ow, h, w, p, sh, sw, hori, norm, mean, stdv, img_data, result);
		}
		else
		{
			uint8_to_float_random_crop_hori_norm_cpu(lengthNew, scale, batch, channels, oh, ow, h, w, p, sh, sw, hori, norm, mean, stdv, img_data, result);
		}
		double moveHeight = static_cast<double>(sh) - static_cast<double>(p);
		double moveWidth = static_cast<double>(sw) - static_cast<double>(p);
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("flip", static_cast<bool>(hori));
        obj.Set("moveHeight", -moveHeight);
        obj.Set("moveWidth", -moveWidth);
        obj.Set("data", outData);

        return obj;
	}

	Napi::Value imgNormalize(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 6)
		{
            throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!IsTypeArray(args[0], napi_uint8_array) || !args[1].IsNumber() || !args[2].IsNumber() ||
			!IsTypeArray(args[3], napi_float32_array) ||
			!IsTypeArray(args[4], napi_float32_array) || !args[5].IsBoolean())
		{
			throw Napi::TypeError::New(env, "Wrong arguments");
		}

		float scale = args[1].ToNumber().FloatValue();
        UINT32 batch = args[2].ToNumber().Uint32Value();

        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();
        UINT32 length = imgU8.ElementLength();
        UINT8* img_data = reinterpret_cast<UINT8*>(imgU8.ArrayBuffer().Data());

        Napi::Float32Array mean_ = args[3].As<Napi::Float32Array>();
        UINT32 lengthM = mean_.ElementLength();
        float* mean = reinterpret_cast<float*>(mean_.ArrayBuffer().Data());
        Napi::Float32Array std_ = args[4].As<Napi::Float32Array>();
        UINT32 lengthS = std_.ElementLength();
        float* stdv = reinterpret_cast<float*>(std_.ArrayBuffer().Data());
        //printf("%d", lengthM);
		if (lengthM != lengthS) {

			throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels != lengthS");
		}

        Napi::Float32Array outData = Napi::Float32Array::New(env, length);
        float* result = (float*)outData.ArrayBuffer().Data();
        bool channelFirst = args[5].ToBoolean().Value();

        if (channelFirst)
        {
            uint8_to_float_convert_norm_o_cpu(length, scale, batch, lengthM, mean, stdv, img_data, result);
        }
        else
        {
            uint8_to_float_convert_norm_cpu(length, scale, batch, lengthM, mean, stdv, img_data, result);
        }

        return outData;
	}

    Napi::Value imgScale(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        if (args.Length() < 7)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!IsTypeArray(args[0], napi_uint8_array) || !args[1].IsNumber() || !args[2].IsNumber() ||
            !args[3].IsNumber() ||
            !args[4].IsNumber() ||
            !args[5].IsNumber() || !args[6].IsBoolean())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        UINT32 oh = args[1].ToNumber().Uint32Value();
        UINT32 ow = args[2].ToNumber().Uint32Value();
        float scaleh = args[3].ToNumber().FloatValue();
        float scalew = args[4].ToNumber().FloatValue();
        UINT32 batch = args[5].ToNumber().Uint32Value();
        bool channelFirst = args[6].ToBoolean().Value();

        int sh = oh * scaleh;
        int sw = ow * scalew;

        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();

        if (sh == oh && ow == sw)
        {
            return imgU8;
        }

        UINT32 length = imgU8.ElementLength();
        UINT32 channels = length / (batch * (oh * ow));
        UINT8* img_data = reinterpret_cast<UINT8*>(imgU8.ArrayBuffer().Data());
        Shape sharp = Shape(batch, channels, oh, ow);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, batch * channels * sh * sw);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        uint8_to_uint8_scale_cpu(length, img_data, sharp, sh, sw, result, channelFirst);

        return outData;
    }

    Napi::Value imgColorHSV(const Napi::CallbackInfo& args)
    {
        Napi::Env env = args.Env();

        if (args.Length() < 7)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }

        if (!IsTypeArray(args[0], napi_uint8_array) || !args[1].IsNumber() || !args[2].IsNumber() ||
            !args[3].IsNumber() ||
            !args[4].IsNumber() ||
            !args[5].IsNumber() ||
            !args[6].IsNumber() || !args[7].IsBoolean())
        {
            throw Napi::TypeError::New(env, "Wrong arguments");
        }

        UINT32 oh = args[1].ToNumber().Uint32Value();
        UINT32 ow = args[2].ToNumber().Uint32Value();
        float hue = args[3].ToNumber().FloatValue();
        float sat = args[4].ToNumber().FloatValue();
        float val = args[5].ToNumber().FloatValue();
        UINT32 batch = args[6].ToNumber().Uint32Value();
        bool channelFirst = args[7].ToBoolean().Value();
        
        Napi::Uint8Array imgU8 = args[0].As<Napi::Uint8Array>();
        
        UINT32 length = imgU8.ElementLength();
        UINT32 pixels = batch * (oh * ow);
        UINT32 channels = length / pixels;
        if (channels < 3 || channels > 4)
        {
            throw Napi::TypeError::New(env, "Wrong number of arguments");
        }
        UINT8* img_data = reinterpret_cast<UINT8*>(imgU8.ArrayBuffer().Data());
        Shape sharp = Shape(batch, channels, oh, ow);
        Napi::Uint8Array outData = Napi::Uint8Array::New(env, length);
        UINT8* result = (UINT8*)outData.ArrayBuffer().Data();

        uint8_to_uint8_color_cpu(pixels, img_data, sharp, hue, sat, val, result, channelFirst);

        return outData;
    }

	Napi::Value test(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();
		float reslut = (float(1) + float(1));

		return Napi::Number::New(env, double(reslut));
	}

    Napi::Object init(Napi::Env env, Napi::Object exports) {
        exports.Set(Napi::String::New(env, "removeAlpha"), Napi::Function::New(env, removeImgBufferAlpha));
        exports.Set(Napi::String::New(env, "captureBBImg"), Napi::Function::New(env, captureImgByBoundingBox));
        exports.Set(Napi::String::New(env, "convertNetData"), Napi::Function::New(env, convertImgDataToNetData));
        exports.Set(Napi::String::New(env, "imgRandomCropHorizontalFlipNormalize"), Napi::Function::New(env, imgRandomCropHorizontalFlipNormalize));
        exports.Set(Napi::String::New(env, "imgNormalize"), Napi::Function::New(env, imgNormalize));
        exports.Set(Napi::String::New(env, "imgScale"), Napi::Function::New(env, imgScale));
        exports.Set(Napi::String::New(env, "imgColorHSV"), Napi::Function::New(env, imgColorHSV));
        exports.Set(Napi::String::New(env, "test"), Napi::Function::New(env, test));

        ImageWrap::Init(env, exports, "Image");
        MateDataWrap::Init(env, exports, "MateData");
        return exports;
    }

    NODE_API_MODULE(NODE_GYP_MODULE_NAME, init)
}

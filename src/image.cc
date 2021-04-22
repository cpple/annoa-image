#include <napi.h>
#include <cmath>
#include <random>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include <string>
#include "../include/util.h"
#include "../include/tool.h"
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
			UINT8 * result = (UINT8*)outData.ArrayBuffer().Data();;

			capture_bbox_img_cpu(picDim, img_data, channels, width, height, x1, y1, Bw, Bh, result);
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

		if (args.Length() < 1)
		{throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!args[0].IsUint8Array() && !args[0].IsUint8ClampedArray())
		{
			throw Napi::TypeError::New(env, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		float scale = 1.0f;
		if (args[1].IsNumber())
		{
			scale = static_cast<float>(args[1].NumberValue(context).ToChecked());
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0].IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0].IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		UINT32 bytes = length * sizeof(float);
		Local<ArrayBuffer> out = ArrayBuffer::New(isolate, bytes);
		Local<Float32Array> outData = Float32Array::New(out, 0, length);
		float * result = (float*)out->GetBackingStore()->Data();

		uint8_to_float_convert_cpu(length, scale, img_data, result);

		args.GetReturnValue().Set(outData);
	}

	Napi::Value imgRandomCropHorizontalFlipNormalize(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 13)
		{throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!args[0].IsUint8Array() && !args[0].IsUint8ClampedArray() || !args[1].IsNumber() || !args[2].IsNumber() || !args[3].IsNumber() ||
			!args[4].IsNumber() || !args[5].IsNumber() || !args[6].IsNumber() || !args[7].IsNumber() || !args[8].IsBoolean() || !args[9].IsBoolean() || !args[10].IsBoolean() ||
			(!args[11].IsNullOrUndefined() && !args[11].IsFloat32Array()) ||
			(!args[12].IsNullOrUndefined() && !args[12].IsFloat32Array()))
		{
			throw Napi::TypeError::New(env, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;
		UINT32 lengthM = 0;
		UINT32 lengthS = 0;
		UINT32 lengthNew = 0;
		UINT32 channels = 0;

		if (args[0].IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0].IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		float scale = static_cast<float>(args[1].NumberValue(context).ToChecked());
		UINT32 batch = static_cast<UINT32>(args[2].NumberValue(context).ToChecked());
		UINT32 oh = static_cast<UINT32>(args[3].NumberValue(context).ToChecked());
		UINT32 ow = static_cast<UINT32>(args[4].NumberValue(context).ToChecked());
		UINT32 h = static_cast<UINT32>(args[5].NumberValue(context).ToChecked());
		UINT32 w = static_cast<UINT32>(args[6].NumberValue(context).ToChecked());
		UINT32 p = static_cast<UINT32>(args[7].NumberValue(context).ToChecked());
		bool channelFirst = args[8].As<v8::Boolean>()->Value();
		bool hori = args[9].As<v8::Boolean>()->Value();
		bool norm = args[10].As<v8::Boolean>()->Value();
		float* mean = nullptr;
		float* stdv = nullptr;
		std::random_device rd;

		if (hori) {
			hori = rd() & 1;
		}

		channels = length / (oh * ow * batch);
		lengthNew = h * w * channels * batch;
		if (norm && args[11].IsFloat32Array() && args[12].IsFloat32Array()) {

			Local<Float32Array> mean_ = Local<Float32Array>::Cast(args[11].ToObject(context).ToLocalChecked());
			lengthM = mean_->Length();
			mean = (float*)mean_->Buffer()->GetBackingStore()->Data();
			Local<Float32Array> std_ = Local<Float32Array>::Cast(args[12].ToObject(context).ToLocalChecked());
			lengthS = std_->Length();
			stdv = (float*)std_->Buffer()->GetBackingStore()->Data();
			if (channels != lengthM || channels != lengthS) {

				throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels != lengthS", NewStringType::kNormal).ToLocalChecked()));
				return;
			}
		}

		if (oh <= p || ow <= p) {

			throw Napi::TypeError::New(env, "Wrong arguments oh + p < h", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (oh + p < h) {

			throw Napi::TypeError::New(env, "Wrong arguments oh + p < h", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (ow + p < w) {

			throw Napi::TypeError::New(env, "Wrong arguments ow + p < w", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (channels < 1 && channels > 4) {

			throw Napi::TypeError::New(env, "Wrong arguments channels < 1 && channels > 4", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		UINT32 sh = rd() % ((oh + 2 * p) - h);
		UINT32 sw = rd() % ((ow + 2 * p) - w);

		UINT32 bytes = length * sizeof(UINT8);
		UINT32 bytesNew = lengthNew * sizeof(float);
		UINT32 bytesM = lengthM * sizeof(float);
		Local<ArrayBuffer> out = ArrayBuffer::New(isolate, bytesNew);
		Local<Float32Array> outData = Float32Array::New(out, 0, lengthNew);
		float* result = (float*)out->GetBackingStore()->Data();

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
		Local<Object> obj = Object::New(isolate);
		obj->Set(context, String::NewFromUtf8(isolate, "flip", NewStringType::kNormal).ToLocalChecked(), Boolean::New(isolate, static_cast<bool>(hori)));
		obj->Set(context, String::NewFromUtf8(isolate, "moveHeight", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, -moveHeight));
		obj->Set(context, String::NewFromUtf8(isolate, "moveWidth", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, -moveWidth));
		obj->Set(context, String::NewFromUtf8(isolate, "data", NewStringType::kNormal).ToLocalChecked(), outData);

		args.GetReturnValue().Set(obj);
	}

	Napi::Value imgNormalize(const Napi::CallbackInfo& args)
	{
        Napi::Env env = args.Env();

		if (args.Length() < 5)
		{throw Napi::TypeError::New(env, "Wrong number of arguments");
		}

		if (!args[0].IsUint8Array() && !args[0].IsUint8ClampedArray() || !args[1].IsNumber() || !args[2].IsNumber() ||
			!args[3].IsFloat32Array() ||
			!args[4].IsFloat32Array())
		{
			throw Napi::TypeError::New(env, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		float scale = static_cast<float>(args[1].NumberValue(context).ToChecked());
		int batch = static_cast<int>(args[2].NumberValue(context).ToChecked());
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0].IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0].IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0].ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		Local<Float32Array> mean_ = Local<Float32Array>::Cast(args[3].ToObject(context).ToLocalChecked());
		int lengthM = mean_->Length();
		const float* mean = (float*)mean_->Buffer()->GetBackingStore()->Data();
		Local<Float32Array> std_ = Local<Float32Array>::Cast(args[4].ToObject(context).ToLocalChecked());
		int lengthS = std_->Length();
		const float* stdv = (float*)std_->Buffer()->GetBackingStore()->Data();
		if (lengthM != lengthS) {

			throw Napi::TypeError::New(env, "Wrong arguments channels != lengthM || channels != lengthS", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		UINT32 bytes = length * sizeof(float);
		UINT32 bytesM = lengthM * sizeof(float);
		Local<ArrayBuffer> out = ArrayBuffer::New(isolate, bytes);
		Local<Float32Array> outData = Float32Array::New(out, 0, length);
		float* result = (float*)out->GetBackingStore()->Data();
		//printf("SolverHandler:%f %d %d \n", scale, batch, lengthM);

		uint8_to_float_convert_norm_cpu(length, scale, batch, lengthM, mean, stdv, img_data, result);

		args.GetReturnValue().Set(outData);
	}

	Napi::Value test(const FunctionCallbackInfo<Value>& args)
	{
        Napi::Env env = args.Env();
		float reslut = (float(1) + float(1));

		args.GetReturnValue().Set(Number::New(isolate, double(reslut)));
	}

    Napi::Value init(Local<Object> exports) {
		NODE_SET_METHOD(exports, "removeAlpha", removeImgBufferAlpha);
		NODE_SET_METHOD(exports, "captureBBImg", captureImgByBoundingBox);
		NODE_SET_METHOD(exports, "convertNetData", convertImgDataToNetData);
        NODE_SET_METHOD(exports, "imgRandomCropHorizontalFlipNormalize", imgRandomCropHorizontalFlipNormalize);
        NODE_SET_METHOD(exports, "imgNormalize", imgNormalize);
		NODE_SET_METHOD(exports, "test", test);
    }
    NODE_MODULE(NODE_GYP_MODULE_NAME, init)
}

// hello.cc
#include <v8.h>
#include <node.h>
#include <cmath>
#include <random>
#include <stdio.h>
#include <iostream>
#include <sstream>

#include <string>
#include "../include/util.h"
#include "../include/tool.h"
//namespace annoa
//{
    using namespace node;
    using namespace v8;
    using v8::Context;
    using v8::Function;
    using v8::FunctionCallbackInfo;
    using v8::FunctionTemplate;
    using v8::Isolate;
    using v8::Local;
    using v8::Number;
    using v8::Object;
    using v8::Persistent;
    using v8::String;
    using v8::Value;

	void removeImgBufferAlpha(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();

		if (args.Length() < 1)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (!args[0]->IsUint8Array() && !args[0]->IsUint8ClampedArray())
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0]->IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0]->IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}
		if (length % 4 != 0)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "img channels error arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		UINT32 bytes = (length / 4) * 3 * sizeof(UINT8);
		UINT32 pix_count = (length / 4);
		Local<ArrayBuffer> out = ArrayBuffer::New(isolate, bytes);
		Local<Uint8Array> outData = Uint8Array::New(out, 0, bytes / sizeof(UINT8));
		UINT8 * result = (UINT8*)out->GetBackingStore()->Data();

		remove_alpha_cpu(pix_count, img_data, result);

		args.GetReturnValue().Set(outData);
	}

	void captureImgByBoundingBox(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();

		if (args.Length() < 4)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if ((!args[0]->IsUint8Array() && !args[0]->IsUint8ClampedArray()) || !args[1]->IsNumber() || !args[2]->IsNumber() || !args[3]->IsNumber() || (!args[4]->IsArray() && !args[4]->IsUint32Array() && !args[4]->IsFloat32Array()))
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0]->IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0]->IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		UINT32 channels = static_cast<UINT32>(args[1]->NumberValue(context).ToChecked());
		UINT32 height = static_cast<UINT32>(args[2]->NumberValue(context).ToChecked());
		UINT32 width = static_cast<UINT32>(args[3]->NumberValue(context).ToChecked());

		if (length % channels != 0 || length != channels * height * width)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "img channels error arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		Local<Array> bboxList = Array::New(isolate);
		if (args[4]->IsUint32Array() || args[4]->IsFloat32Array())
		{
			Local<TypedArray> bbox = Local<TypedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
			bboxList->Set(context, bboxList->Length(), bbox);
		}
		if (args[4]->IsArray())
		{
			bboxList = Local<Array>::Cast(args[4]->ToObject(context).ToLocalChecked());
		}
		Local<Array> imgList = Array::New(isolate);
		UINT32 size = bboxList->Length();
		for (UINT32 c = 0; c < size; c++) {

			Local<Value> bboxV = bboxList->Get(context, c).ToLocalChecked();
			if (!bboxV->IsUint32Array() && !bboxV->IsFloat32Array())
			{
				isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "bbox data type error arguments", NewStringType::kNormal).ToLocalChecked()));
				return;
			}
			bool isFloat = false;
			if (bboxV->IsFloat32Array())
			{
				isFloat = true;
			}
			Local<TypedArray> bbox = Local<TypedArray>::Cast(bboxList->Get(context, c).ToLocalChecked());
			if (bbox->Length() != 4)
			{
				isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "bbox length error arguments", NewStringType::kNormal).ToLocalChecked()));
				return;
			}
			void* data = bbox->Buffer()->GetBackingStore()->Data();

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

			Local<ArrayBuffer> out = ArrayBuffer::New(isolate, spDim * sizeof(UINT8));
			Local<Uint8Array> outData = Uint8Array::New(out, 0, spDim);
			UINT8 * result = (UINT8*)out->GetBackingStore()->Data();

			capture_bbox_img_cpu(picDim, img_data, channels, width, height, x1, y1, Bw, Bh, result);
			Local<Object> obj = Object::New(isolate);
			obj->Set(context, String::NewFromUtf8(isolate, "width", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, static_cast<double>(Bw)));
			obj->Set(context, String::NewFromUtf8(isolate, "height", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, static_cast<double>(Bh)));
			obj->Set(context, String::NewFromUtf8(isolate, "channels", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, static_cast<double>(channels)));
			obj->Set(context, String::NewFromUtf8(isolate, "data", NewStringType::kNormal).ToLocalChecked(), outData);
			imgList->Set(context, imgList->Length(), obj);
		}
		args.GetReturnValue().Set(imgList);
	}

	void convertImgDataToNetData(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();

		if (args.Length() < 1)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (!args[0]->IsUint8Array() && !args[0]->IsUint8ClampedArray())
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		float scale = 1.0f;
		if (args[1]->IsNumber())
		{
			scale = static_cast<float>(args[1]->NumberValue(context).ToChecked());
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0]->IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0]->IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
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

	void imgRandomCropHorizontalFlipNormalize(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();

		if (args.Length() < 13)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (!args[0]->IsUint8Array() && !args[0]->IsUint8ClampedArray() || !args[1]->IsNumber() || !args[2]->IsNumber() || !args[3]->IsNumber() ||
			!args[4]->IsNumber() || !args[5]->IsNumber() || !args[6]->IsNumber() || !args[7]->IsNumber() || !args[8]->IsBoolean() || !args[9]->IsBoolean() || !args[10]->IsBoolean() ||
			(!args[11]->IsNullOrUndefined() && !args[11]->IsFloat32Array()) ||
			(!args[12]->IsNullOrUndefined() && !args[12]->IsFloat32Array()))
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}
		UINT8* img_data = nullptr;
		UINT32 length = 0;
		UINT32 lengthM = 0;
		UINT32 lengthS = 0;
		UINT32 lengthNew = 0;
		UINT32 channels = 0;

		if (args[0]->IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0]->IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		float scale = static_cast<float>(args[1]->NumberValue(context).ToChecked());
		UINT32 batch = static_cast<UINT32>(args[2]->NumberValue(context).ToChecked());
		UINT32 oh = static_cast<UINT32>(args[3]->NumberValue(context).ToChecked());
		UINT32 ow = static_cast<UINT32>(args[4]->NumberValue(context).ToChecked());
		UINT32 h = static_cast<UINT32>(args[5]->NumberValue(context).ToChecked());
		UINT32 w = static_cast<UINT32>(args[6]->NumberValue(context).ToChecked());
		UINT32 p = static_cast<UINT32>(args[7]->NumberValue(context).ToChecked());
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
		if (norm && args[11]->IsFloat32Array() && args[12]->IsFloat32Array()) {

			Local<Float32Array> mean_ = Local<Float32Array>::Cast(args[11]->ToObject(context).ToLocalChecked());
			lengthM = mean_->Length();
			mean = (float*)mean_->Buffer()->GetBackingStore()->Data();
			Local<Float32Array> std_ = Local<Float32Array>::Cast(args[12]->ToObject(context).ToLocalChecked());
			lengthS = std_->Length();
			stdv = (float*)std_->Buffer()->GetBackingStore()->Data();
			if (channels != lengthM || channels != lengthS) {

				isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments channels != lengthM || channels != lengthS", NewStringType::kNormal).ToLocalChecked()));
				return;
			}
		}

		if (oh <= p || ow <= p) {

			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments oh + p < h", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (oh + p < h) {

			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments oh + p < h", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (ow + p < w) {

			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments ow + p < w", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (channels < 1 && channels > 4) {

			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments channels < 1 && channels > 4", NewStringType::kNormal).ToLocalChecked()));
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

	void imgNormalize(const v8::FunctionCallbackInfo<v8::Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();

		if (args.Length() < 5)
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong number of arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		if (!args[0]->IsUint8Array() && !args[0]->IsUint8ClampedArray() || !args[1]->IsNumber() || !args[2]->IsNumber() ||
			!args[3]->IsFloat32Array() ||
			!args[4]->IsFloat32Array())
		{
			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments", NewStringType::kNormal).ToLocalChecked()));
			return;
		}

		float scale = static_cast<float>(args[1]->NumberValue(context).ToChecked());
		int batch = static_cast<int>(args[2]->NumberValue(context).ToChecked());
		UINT8* img_data = nullptr;
		UINT32 length = 0;

		if (args[0]->IsUint8Array())
		{
			Local<Uint8Array> imgU8 = Local<Uint8Array>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgU8->Length();
			img_data = (UINT8*)imgU8->Buffer()->GetBackingStore()->Data();
		}

		if (args[0]->IsUint8ClampedArray())
		{
			Local<Uint8ClampedArray> imgUC8 = Local<Uint8ClampedArray>::Cast(args[0]->ToObject(context).ToLocalChecked());
			length = imgUC8->Length();
			img_data = (UINT8*)imgUC8->Buffer()->GetBackingStore()->Data();
		}

		Local<Float32Array> mean_ = Local<Float32Array>::Cast(args[3]->ToObject(context).ToLocalChecked());
		int lengthM = mean_->Length();
		const float* mean = (float*)mean_->Buffer()->GetBackingStore()->Data();
		Local<Float32Array> std_ = Local<Float32Array>::Cast(args[4]->ToObject(context).ToLocalChecked());
		int lengthS = std_->Length();
		const float* stdv = (float*)std_->Buffer()->GetBackingStore()->Data();
		if (lengthM != lengthS) {

			isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Wrong arguments channels != lengthM || channels != lengthS", NewStringType::kNormal).ToLocalChecked()));
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

	void test(const FunctionCallbackInfo<Value>& args)
	{
		Isolate* isolate = args.GetIsolate();
		Local<Context> context = isolate->GetCurrentContext();
		float reslut = (float(1) + float(1));

		args.GetReturnValue().Set(Number::New(isolate, double(reslut)));
	}

    void init(Local<Object> exports) {
		NODE_SET_METHOD(exports, "removeAlpha", removeImgBufferAlpha);
		NODE_SET_METHOD(exports, "captureBBImg", captureImgByBoundingBox);
		NODE_SET_METHOD(exports, "convertNetData", convertImgDataToNetData);
        NODE_SET_METHOD(exports, "imgRandomCropHorizontalFlipNormalize", imgRandomCropHorizontalFlipNormalize);
        NODE_SET_METHOD(exports, "imgNormalize", imgNormalize);
		NODE_SET_METHOD(exports, "test", test);
    }
    NODE_MODULE(NODE_GYP_MODULE_NAME, init)
//}
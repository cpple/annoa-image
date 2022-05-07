#ifndef IMAGE_H_
#define IMAGE_H_
#include <memory>
#include <string>
#include <vector>
#include "../../include/cuda/cuda.h"
#include "../../include/napi.h"
#include "../../include/util.h"
#include "../../include/tool.h"

namespace annoa
{
    class ImageWrap : public Napi::ObjectWrap<ImageWrap> {
    public:
        static Napi::Object Init(Napi::Env env, Napi::Object exports, std::string keyname);
        ImageWrap(const Napi::CallbackInfo& info);
        ~ImageWrap();
        void ImageWrap::Destructor(napi_env env, void* nativeObject, void* /*finalize_hint*/);
        static Napi::Value Create(const Napi::Number& channel, const Napi::Number& height, const Napi::Number& width);
    private:
        Napi::Value GetNCHWData(const Napi::CallbackInfo& info);
        Napi::Value GetNHWCData(const Napi::CallbackInfo& info);
        Napi::Value SetNCHWData(const Napi::CallbackInfo& info);
        Napi::Value SetNHWCData(const Napi::CallbackInfo& info);
        Napi::Value RemoveAlpha(const Napi::CallbackInfo& info);
        Napi::Value HorizontalFlip(const Napi::CallbackInfo& info);
        Napi::Value RandomCrop(const Napi::CallbackInfo& info);
        Napi::Value Normalize(const Napi::CallbackInfo& info);
        Napi::Value MateData(const Napi::CallbackInfo& info);
        Napi::Value ScaleSize(const Napi::CallbackInfo& info);
        Napi::Value ColorHSV(const Napi::CallbackInfo& args);
        Napi::Value CaptureImgByBoundingBox(const Napi::CallbackInfo& args);

        Napi::Value ScaleSizeGPU(const Napi::CallbackInfo& info);
    public:
        void SetData(int flag, Napi::Uint8Array& array_);
    private:
        static Napi::FunctionReference constructor;
        Shape _shape;
        void* _data = nullptr;
        INT8 _flag = 0;
    };
}
#endif

#ifndef MATEDATA_H_
#define MATEDATA_H_
#include <memory>
#include <string>
#include <vector>
#include "../../include/NAPI.h"
#include "../../include/util.h"
#include "../../include/tool.h"

namespace annoa
{
    class MateDataWrap : public Napi::ObjectWrap<MateDataWrap> {
    public:
        static Napi::Object Init(Napi::Env env, Napi::Object exports, std::string keyname);
        MateDataWrap(const Napi::CallbackInfo& info);
        ~MateDataWrap();
        void MateDataWrap::Destructor(napi_env env, void* nativeObject, void* /*finalize_hint*/);
        static Napi::Value Create(const Napi::Number& channel, const Napi::Number& height, const Napi::Number& width);
    private:
        Napi::Value GetData(const Napi::CallbackInfo& info);
        Napi::Value SetData(const Napi::CallbackInfo& info);
        Napi::Value Normalize(const Napi::CallbackInfo& info);
        Napi::Value Scale(const Napi::CallbackInfo& info);

    public:
        void SetData(int flag, Napi::Float32Array& data);

        inline void set_data(void* data) {
            _data = data;
        };
        inline void set_batch(int number) {
            _shape.n = number;
        };

    private:
        static Napi::FunctionReference constructor;
        Shape _shape;
        void* _data = nullptr;
        INT8 _flag = 0;
    };
}
#endif

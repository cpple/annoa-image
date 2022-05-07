#pragma once
#ifndef CUDA_T_H_
#define CUDA_T_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <curand_kernel.h>
#include <curand.h>

#include <cudnn.h>

//#include <nccl.h>
#include <cublas_v2.h>

#include "./CudaDevice.h"
//cpu

#define CUDA_UTIL_HD __host__ __device__
#define CUDA_UTIL_IHD __inline__ __host__ __device__


inline void FatalError(const std::string& s) {
    std::cerr << s << "\nAborting...\n";
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

inline void checkCudaErrors(cudaError_t status) {
    if (status != 0) {
        std::stringstream _error;
        _error << "Cuda failure\nError: " << cudaGetErrorString(status);
        FatalError(_error.str());
    }
}

class AnnoaCuda {

public:
    static int maxThreadsPerBlock;

    static CudaTool* cuda_tool;

    static void init() {
        AnnoaCuda::maxThreadsPerBlock = CudaDevice::setOptimalThreadsPerBlock();
        AnnoaCuda::cuda_tool = &CudaTool::Get();
    }
    static cudaStream_t Stream() {
        return cuda_tool->Stream_T();
    }
    static cudaStream_t RandStream() {
        return cuda_tool->RandStream_T();
    }
    static void* AnnoaMallocCopyDevice(size_t byte, const void* ptr = nullptr) {

        void* gpu_ptr_ = nullptr;
        checkCudaErrors(cudaMalloc(&gpu_ptr_, byte));
        if (ptr) {
            const cudaMemcpyKind put = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy(gpu_ptr_, ptr, byte, put));
        }
        return gpu_ptr_;
    }
    static void AnnoaDeviceCopyHost(const void* gpu_ptr, void* cpu_ptr, size_t byte) {

        const cudaMemcpyKind put = cudaMemcpyDeviceToHost;
        checkCudaErrors(cudaMemcpy(cpu_ptr, gpu_ptr, byte, put));
    }
    static void AnnoaFreeMemDevice(void* gpu_ptr_) {

        checkCudaErrors(cudaFree(gpu_ptr_));
        gpu_ptr_ = nullptr;
    }
    template <typename Dtype>
    CUDA_UTIL_IHD Dtype max_dtype();
    template <>
    CUDA_UTIL_IHD double max_dtype<double>() {
        return DBL_MAX;
    }
    template <>
    CUDA_UTIL_IHD float max_dtype<float>() {
        return FLT_MAX;
    }
    template <typename Dtype>
    CUDA_UTIL_IHD Dtype min_dtype();
    template <>
    CUDA_UTIL_IHD double min_dtype<double>() {
        return DBL_MIN;
    }
    template <>
    CUDA_UTIL_IHD float min_dtype<float>() {
        return FLT_MIN;
    }
    template <typename Dtype>
    CUDA_UTIL_IHD Dtype epsilon_dtype();
    template <>
    CUDA_UTIL_IHD double epsilon_dtype<double>() {
        return DBL_EPSILON;
    }
    template <>
    CUDA_UTIL_IHD float epsilon_dtype<float>() {
        return FLT_EPSILON;
    }
};

#define MAX_TREADS_PER_BLOCK AnnoaCuda::maxThreadsPerBlock

    inline int NUM_BLOCKS(const int N) {
        return (N + MAX_TREADS_PER_BLOCK - 1) / MAX_TREADS_PER_BLOCK;
    }

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK checkCudaErrors(cudaPeekAtLastError())

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#if !defined(CUDNN_VERSION) || !CUDNN_VERSION_MIN(6, 0, 0)
#error "annoa 0.0.1 and higher requires CuDNN version 6.0.0 or higher"
#endif

#endif

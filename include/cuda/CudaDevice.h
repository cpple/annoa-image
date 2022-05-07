#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>

class CudaDevice {
public:
	static int setOptimalThreadsPerBlock();
	static size_t numOfBlocks(const size_t &size);
	static int maxThreadsPerBlock;
	static bool inited;
protected:
	static std::vector<cudaDeviceProp> _cudaDeviceProps;
};

inline void printDeviceProp(const cudaDeviceProp& prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %zd.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %zd.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %zd.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %zd.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %zd.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

class CudaTool {
public:
	~CudaTool();
	void init();
    static CudaTool& Get();
	inline cudaStream_t Stream_T() { return stream_; };
    inline cudaStream_t RandStream_T() { return stream_rand_; };
private:
	cudaStream_t stream_;
	cudaStream_t stream_rand_;
private:
	CudaTool();
};

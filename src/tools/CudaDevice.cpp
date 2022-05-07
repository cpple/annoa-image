#pragma once
#include "../../include/cuda/CudaDevice.h"
#include "../../include/cuda/cuda.h"

int CudaDevice::setOptimalThreadsPerBlock() {
	int nDevices;
	int threadsNum = 0;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//printDeviceProp(prop);
		//maxThreadsPerBlock = prop.maxThreadsPerBlock;
		threadsNum += maxThreadsPerBlock;
	}
	inited = true;
	return maxThreadsPerBlock;
    printf("nDevices %d\nthreadsNum %d\n", nDevices, threadsNum);
}

size_t CudaDevice::numOfBlocks(const size_t &size) {
	return (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
}

int CudaDevice::maxThreadsPerBlock = 512;
bool CudaDevice::inited = false;

CudaTool::CudaTool() {
    init();
}
CudaTool::~CudaTool() {
    printf("fuck you!!!!!!!!!");
    cudaStreamDestroy(stream_);
    cudaStreamDestroy(stream_rand_);
}
CudaTool& CudaTool::Get(){
  static CudaTool tool;
  return tool;
}
void CudaTool::init(){
	checkCudaErrors(cudaStreamCreate(&stream_));
	checkCudaErrors(cudaStreamCreate(&stream_rand_));
}

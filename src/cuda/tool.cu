#include <algorithm>
#include <time.h>
#include <cmath>
#include <random>
#include "../../include/cuda/cuda.h"
#include "../../include/tool.h"
#include "../../include/util.h"

__device__ float get_value_(
    const UINT8* data, const int number, const int channel, const int height, const int width,
    const unsigned int n, const unsigned int c,
    int y, int x, const bool& cf
) {
    // Replicate border for 1 pixel
    if (x == -1) x = 0;
    if (x == width) x = width - 1;
    if (y == -1) y = 0;
    if (y == height) y = height - 1;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        // N*cs*hs*ws + C*hs*ws + H*ws + W
        if (cf)
        {
            return float(data[(n * height * width * channel) +
                (y * width * channel) + (x * channel) + c]);
        }
        else
        {
            return float(data[(n * channel * height * width) +
                (c * height * width) + (y * width) + x]);
        }
    }
    else {
        return float(0);
    }
}

__device__ float cubic_interpolation_(const float& d,
    const float& v1, const float& v2, const float& v3, const float& v4
) {
    // d is [0,1], marking the distance from v2 towards v3
    return v2 + d * (
        -2.0 * v1 - 3.0 * v2 + 6.0 * v3 - 1.0 * v4 + d * (
            3.0 * v1 - 6.0 * v2 + 3.0 * v3 + 0.0 * v4 + d * (
                -1.0 * v1 + 3.0 * v2 - 3.0 * v3 + 1.0 * v4))) / 6.0;
}


// Interpolate in 1D space
__device__ float interpolate_x_(
    const UINT8* data, const int number, const int channel, const int height, const int width,
    const unsigned int n, const unsigned int c,
    const int y, const float x, const bool& cf
) {
    float dx = x - floor(x);
    return cubic_interpolation_(dx,
        get_value_(data, number, channel, height, width, n, c, y, floor(x) - 1, cf),
        get_value_(data, number, channel, height, width, n, c, y, floor(x), cf),
        get_value_(data, number, channel, height, width, n, c, y, ceil(x), cf),
        get_value_(data, number, channel, height, width, n, c, y, ceil(x) + 1, cf));
}


// Interpolate in 2D space
__device__ float interpolate_xy_(
    const UINT8* data, const int number, const int channel, const int height, const int width,
    const unsigned int n, const unsigned int c,
    const float y, const float x, const bool& cf
) {
    float dy = y - floor(y);
    float v = cubic_interpolation_(dy,
        interpolate_x_(data, number, channel, height, width, n, c, floor(y) - 1, x, cf),
        interpolate_x_(data, number, channel, height, width, n, c, floor(y), x, cf),
        interpolate_x_(data, number, channel, height, width, n, c, ceil(y), x, cf),
        interpolate_x_(data, number, channel, height, width, n, c, ceil(y) + 1, x, cf));
    if (v < 0)
    {
        v = 0;
    }
    if (v > 255)
    {
        v = 255;
    }
    return v;
}

__global__ void uint8_to_uint8_scale_kernel_gpu_(const int n, const UINT8* a, const int number, const int channel, const int height, const int width, const int sh, const int sw, UINT8* out) {
    CUDA_KERNEL_LOOP(index, n) {

        int spDim = sh * sw;
        //int bDim = spDim;// *shape.c;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);
        //int channels = tmpBIdx / spDim;
        //int tmpCIdx = tmpBIdx % spDim;

        //float ph = (float(shape.h) - sh) / float(2.0);
        //float pw = (float(shape.w) - sw) / float(2.0);

        int dst_x = (tmpBIdx / sw);
        int dst_y = (tmpBIdx % sw);

        float x = static_cast<float>(dst_x);
        float y = static_cast<float>(dst_y);
        // scale
        if (width != sw) {
            x *= float(width - 1) / float(sw - 1);
        }
        if (height != sh) {
            y *= float(height - 1) / float(sh - 1);
        }
        for (int c = 0; c < channel; c++) {
            // N*cs*hs*ws + C*hs*ws + H*ws + W
            const int dst_idx = (batch * channel * sh * sw) +
                (c * sh * sw) + (dst_y * sw) + int(dst_x);
            UINT8 V = static_cast<UINT8>(interpolate_xy_(a, number, channel, height, width, batch, c, y, x, false));
            out[dst_idx] = V;
        }
    }
}

__global__ void uint8_to_uint8_scale_o_kernel_gpu_(const int n, const UINT8* a, const int number, const int channel, const int height, const int width, const int sh, const int sw, UINT8* out) {
    CUDA_KERNEL_LOOP(index, n) {

        int spDim = sh * sw;
        //int bDim = spDim;// *shape.c;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);
        //int channels = tmpBIdx / spDim;
        //int tmpCIdx = tmpBIdx % spDim;

        //float ph = (float(shape.h) - sh) / float(2.0);
        //float pw = (float(shape.w) - sw) / float(2.0);

        int dst_x = (tmpBIdx / sw);
        int dst_y = (tmpBIdx % sw);

        float x = static_cast<float>(dst_x);
        float y = static_cast<float>(dst_y);
        // scale
        if (width != sw) {
            x *= float(width - 1) / float(sw - 1);
        }
        if (height != sh) {
            y *= float(height - 1) / float(sh - 1);
        }
        //printf("sdfhsfjhsjkdfhskjdfhsjkd_0000\n");
        for (int c = 0; c < channel; c++) {
            const int dst_idx = (batch * sh * sw * channel) +
                (dst_y * sw * channel) + (dst_x * channel) + int(c);

            UINT8 V = static_cast<UINT8>(interpolate_xy_(a, number, channel, height, width, batch, c, y, x, true));
            out[dst_idx] = V;
            //printf("sex_0 %d\n", out[dst_idx]);
        }
    }
}

void uint8_to_uint8_scale_gpu(const int N, const UINT8* a, const Shape& shape, const int sh, const int sw, UINT8* y, const bool& cf) {

    const int n = shape.n;
    const int c = shape.c;
    const int h = shape.h;
    const int w = shape.w;
    if (cf)
    {
        uint8_to_uint8_scale_o_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, a, n, c, h, w, sh, sw, y);
    }
    else
    {
        uint8_to_uint8_scale_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, a, n, c, h, w, sh, sw, y);
    }
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

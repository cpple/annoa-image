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

__global__ void random_crop_kernel_gpu_(const int n, const int c, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, int* m) {

    CUDA_KERNEL_LOOP(index, n) {

        int spDim = w * h;
        int bDim = w * h * c;

        int batch = index / bDim;

        int tmpBIdx = (index % bDim);
        int channels = tmpBIdx / spDim;
        int tmpCIdx = (tmpBIdx % spDim);
        int outH = tmpCIdx / w;
        int outW = tmpCIdx % w;

        int inputH = outH;
        int inputW = outW;
        UINT8 v = 0;

        int moveH = rh[batch];
        int moveW = rw[batch];

        inputH = moveH + outH;
        inputW = moveW + outW;
        int spIDim = ow * oh;
        int bIDim = ow * oh * c;

        if (inputH >= 0 && inputH < (int)oh && inputW >= 0 && inputW < (int)ow) {

            v = a[batch * bIDim + channels * spIDim + inputH * (int)ow + inputW];
        }
        y[index] = v;
    }
}

void random_crop_gpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, int* m) {
    // NOLINT_NEXT_LINE(whitespace/operators)

    random_crop_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, channels, oh, ow, h, w, p, rh, rw, a, y, m);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

__global__ void random_crop_nhwc_kernel_gpu_(const int n, const int c, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, int* m) {

    CUDA_KERNEL_LOOP(index, n) {

        //int spDim = w * h;
        int bDim = w * h * c;

        int batch = index / bDim;
        
        int tmpBIdx = (index % bDim);

        int channels = tmpBIdx % c;

        int tmpCIdx = (tmpBIdx / c);

        int outH = tmpCIdx / w;

        int outW = tmpCIdx % w;

        int inputH = outH;
        int inputW = outW;
        UINT8 v = 0;

        int moveH = rh[batch];
        int moveW = rw[batch];

        inputH = moveH + outH;
        inputW = moveW + outW;
        //int spIDim = ow * oh;
        int bIDim = ow * oh * c;

        if (inputH >= 0 && inputH < (int)oh && inputW >= 0 && inputW < (int)ow) {

            v = a[batch * bIDim + inputH * (int)ow * c  + inputW * c + channels];
        }
        y[index] = v;
    }
}

void random_crop_nhwc_gpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, int* m) {
    // NOLINT_NEXT_LINE(whitespace/operators)

    random_crop_nhwc_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, channels, oh, ow, h, w, p, rh, rw, a, y, m);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}


__global__ void horizontal_flip_kernel_gpu_(const int n, const int c, const int h, const int w, UINT8* y) {
    CUDA_KERNEL_LOOP(index, n) {
        int pic_idx = index % (h * w);
        int mini_size = c * h * w;
        int b_ = index / (h * w);
        int w_ = pic_idx % w;
        int w_o = w - w_ - 1;
        int h_ = pic_idx / w;
        if (w_ <= w / 2) {

            for (int c_ = 0; c_ < c; c_++) {
                UINT8 pix = y[b_ * mini_size + c_ * h * w + h_ * w + w_o];
                y[b_ * mini_size + c_ * h * w + h_ * w + w_o] = y[b_ * mini_size + c_ * h * w + h_ * w + w_];
                y[b_ * mini_size + c_ * h * w + h_ * w + w_] = pix;
            }
        }
    }
}

void horizontal_flip_gpu(const int N, const int c, const int h, const int w, UINT8* y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    horizontal_flip_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, c, h, w, y);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

__global__ void horizontal_flip_nhwc_kernel_gpu_(const int n, const int c, const int h, const int w, UINT8* y) {
    CUDA_KERNEL_LOOP(index, n) {
        int pic_idx = index % (h * w);
        int mini_size = c * h * w;
        int b_ = index / (h * w);
        int w_ = pic_idx % w;
        int w_o = w - w_ - 1;
        int h_ = pic_idx / w;
        if (w_ <= w / 2) {

            for (int c_ = 0; c_ < c; c_++) {
                //UINT8 pix = y[index];
                //y[index] = a[b_ * mini_size + c_ * h * w + h_ * w + w_];
                //std::swap(y[b_ * mini_size + h_ * w * c + w_ * c + c_], y[b_ * mini_size + h_ * w * c + w_o * c + c_]);

                UINT8 pix = y[b_ * mini_size + h_ * w * c + w_ * c + c_];
                y[b_ * mini_size + h_ * w * c + w_ * c + c_] = y[b_ * mini_size + h_ * w * c + w_o * c + c_];
                y[b_ * mini_size + h_ * w * c + w_o * c + c_] = pix;
            }
        }
    }
}

void horizontal_flip_nhwc_gpu(const int N, const int c, const int h, const int w, UINT8* y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    horizontal_flip_nhwc_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, c, h, w, y);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

__device__ void convert_rgb_to_hsv_gpu(float r, float g, float b, float* h, float* s, float* v) {
    float min_v = fmin(fmin(r, g), b);
    float max_v = fmax(fmax(r, g), b);
    float delta = max_v - min_v;

    if (max_v == 0 || delta == 0) {
        *h = 0;
        *s = 0;
        *v = max_v;
        return;
    }

    if (r == max_v) {
        *h = (g - b) / delta;
    }
    else if (g == max_v) {
        *h = 2 + (b - r) / delta;
    }
    else {
        *h = 4 + (r - g) / delta;
    }

    *h *= 60;
    if (h < 0) {
        *h += 360;
    }
    *s = delta / max_v;
    *v = max_v;
}

__device__ void convert_hsv_to_rgb_gpu(
    float h, float s, float v,
    float* r, float* g, float* b
) {
    int i;
    float f, p, q, t;

    if (s == 0) {
        *r = v;
        *g = v;
        *b = v;
        return;
    }

    h /= 60;  // sector 0 to 5
    i = floor(h);
    f = h - i;  // factorial part of h
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch (i) {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    default:  // case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    }
}

__global__ void uint8_to_uint8_color_kernel_gpu_(const int n, const UINT8* src_data, const int number, const int channels, const int height, const int width, const float hue, const float sat, const float val, UINT8* dst_data) {
    CUDA_KERNEL_LOOP(index, n) {

        int spDim = height * width;
        int bDim = spDim * channels;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);

        float y = float(tmpBIdx / width);
        float x = float(tmpBIdx % width);

        const bool doHueRotation = (abs(hue) > FLT_EPSILON);
        const bool doDesaturation = (sat < (1.0 - 1.0 / UINT8_MAX));
        const bool doValue = (val >= 0 && val != 1);

        int bIdx = batch * bDim + y * width + x;

        const int channel_stride = height * width;
        float norm = float(1) / float(255);
        // read
        float r = src_data[bIdx + 0 * channel_stride] * norm;
        float g = src_data[bIdx + 1 * channel_stride] * norm;
        float b = src_data[bIdx + 2 * channel_stride] * norm;

        if (doHueRotation || doDesaturation || doValue) {
            // transform
            float h, s, v;
            convert_rgb_to_hsv_gpu(r, g, b, &h, &s, &v);
            if (doHueRotation) {
                h -= hue;
            }
            if (doDesaturation) {
                s *= sat;
            }
            if (doValue) {
                v *= val;
            }
            convert_hsv_to_rgb_gpu(h, s, v, &r, &g, &b);
        }
        UINT8 r8, g8, b8;
        r = round(r / norm);
        g = round(g / norm);
        b = round(b / norm);
        r8 = r > 255. ? UINT8(255) : (r < 0. ? UINT8(0) : static_cast<UINT8>(r));
        g8 = g > 255. ? UINT8(255) : (g < 0. ? UINT8(0) : static_cast<UINT8>(g));
        b8 = b > 255. ? UINT8(255) : (b < 0. ? UINT8(0) : static_cast<UINT8>(b));
        // write
        dst_data[bIdx + 0 * channel_stride] = r8;
        dst_data[bIdx + 1 * channel_stride] = g8;
        dst_data[bIdx + 2 * channel_stride] = b8;
        for (int c = 3; c < channels; c++) {
            dst_data[bIdx + c * channel_stride] = src_data[bIdx + c * channel_stride];
        }
    }
}

__global__ void uint8_to_uint8_color_o_kernel_gpu_(const int n, const UINT8* src_data, const int number, const int channels, const int height, const int width, const float hue, const float sat, const float val, UINT8* dst_data) {

    CUDA_KERNEL_LOOP(index, n) {

        int spDim = height * width;
        int bDim = spDim * channels;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);

        float y = float(tmpBIdx / width);
        float x = float(tmpBIdx % width);

        const bool doHueRotation = (abs(hue) > FLT_EPSILON);
        const bool doDesaturation = (sat < (1.0 - 1.0 / UINT8_MAX));
        const bool doValue = (val >= 0 && val != 1);

        int bIdx = batch * bDim + y * channels * width + x * channels;
        float norm = float(1) / float(255);

        // read
        float r = src_data[bIdx + 0] * norm;
        float g = src_data[bIdx + 1] * norm;
        float b = src_data[bIdx + 2] * norm;

        if (doHueRotation || doDesaturation || doValue) {
            // transform
            float h, s, v;
            convert_rgb_to_hsv_gpu(r, g, b, &h, &s, &v);
            if (doHueRotation) {
                h -= hue;
            }
            if (doDesaturation) {
                s *= sat;
            }
            if (doValue) {
                v *= val;
            }
            convert_hsv_to_rgb_gpu(h, s, v, &r, &g, &b);
        }

        UINT8 r8, g8, b8;
        r = round(r / norm);
        g = round(g / norm);
        b = round(b / norm);
        r8 = r > 255. ? UINT8(255) : (r < 0. ? UINT8(0) : static_cast<UINT8>(r));
        g8 = g > 255. ? UINT8(255) : (g < 0. ? UINT8(0) : static_cast<UINT8>(g));
        b8 = b > 255. ? UINT8(255) : (b < 0. ? UINT8(0) : static_cast<UINT8>(b));
        // write
        dst_data[bIdx + 0] = r8;
        dst_data[bIdx + 1] = g8;
        dst_data[bIdx + 2] = b8;
        for (int c = 3; c < channels; c++) {
            dst_data[bIdx + c] = src_data[bIdx + c];
        }
    }
}

void uint8_to_uint8_color_gpu(const int N, const UINT8* a, const Shape& shape, const float hue, const float sat, const float val, UINT8* y, const bool& cf) {
    //https://www.cs.rit.edu/~ncs/color/t_convert.html
    const int number = shape.n;
    const int channels = shape.c;
    const int height = shape.h;
    const int width = shape.w;
    if (cf)
    {
        uint8_to_uint8_color_o_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, a, number, channels, height, width, hue, sat, val, y);
    }
    else
    {
        uint8_to_uint8_color_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, a, number, channels, height, width, hue, sat, val, y);
    }

    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

__global__ void uint8_to_float_convert_norm_kernel_gpu_(const int n, const float scale, const UINT32 b, const UINT32 c, const float* m, const float* s, const UINT8* a, float* y) {
    CUDA_KERNEL_LOOP(index, n) {

        int spDim = n / (b * c);
        int bDim = spDim * c;

        int tmpBIdx = (index % bDim);
        int channels = tmpBIdx / spDim;
        float v = static_cast<float>(a[index]);
        v *= scale;
        v = (v - m[channels]) / s[channels];
        y[index] = v;
    }
}

void uint8_to_float_convert_norm_gpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y) {

    uint8_to_float_convert_norm_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, scale, batch, channels, m, s, a, y);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

__global__ void uint8_to_float_convert_norm_o_kernel_gpu_(const int n, const float scale, const int b, const int c, const float* m, const float* s, const UINT8* a, float* y) {
    CUDA_KERNEL_LOOP(index, n) {

        int channels = index % c;
        float v = static_cast<float>(a[index]);
        v *= scale;
        v = (v - m[channels]) / s[channels];
        y[index] = v;
    }
}

void uint8_to_float_convert_norm_o_gpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y) {

    uint8_to_float_convert_norm_o_kernel_gpu_ << <NUM_BLOCKS(N), MAX_TREADS_PER_BLOCK, 0, AnnoaCuda::Stream() >> > (N, scale, batch, channels, m, s, a, y);
    CUDA_POST_KERNEL_CHECK;
    checkCudaErrors(cudaStreamSynchronize(AnnoaCuda::Stream()));
}

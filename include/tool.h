#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./cuda/cuda.h"
#include "./util.h"

void remove_alpha_cpu(const int N, const int c, const UINT8* a, UINT8* y);
void remove_alpha_chw_cpu(const int N, const int dima, const int dimy, const UINT8* a, UINT8* y);
void draw_board_by_point(const int N, const int c, const int bh, const int bw, const int ch, const int cw, const UINT8* board, const UINT8* chess, const UINT32* point, const UINT8* idx, UINT8* image);
void uint8_to_float_convert_cpu(const int N, const float scale, const UINT8* a, float* y);
void uint8_to_float_convert_norm_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y);
void uint8_to_float_convert_norm_o_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y);
void capture_bbox_img_cpu(const int N, const UINT8* a, int c, int aw, int ah, int x, int y, int w, int h, UINT8* r, const bool& flag);
void uint8_to_float_random_crop_hori_norm_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y);
void uint8_to_float_random_crop_hori_norm_o_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y);
void uint8_to_uint8_scale_cpu(const int N, const UINT8* a, const Shape& shape, const int sh, const int sw, UINT8* y, const bool& cf);
void uint8_to_uint8_color_cpu(const int N, const UINT8* a, const Shape& shape, const float hue, const float sat, const float val, UINT8* y, const bool& cf);
void nhwc_to_nchw_cpu(const int N, const int c, const int h, const int w, const UINT8* a, UINT8* y);
void nchw_to_nhwc_cpu(const int N, const int c, const int h, const int w, const UINT8* a, UINT8* y);
void horizontal_flip_cpu(const int N, const int c, const int h, const int w, UINT8* y);
void horizontal_flip_nhwc_cpu(const int N, const int c, const int h, const int w, UINT8* y);
void random_crop_cpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const UINT8* a, UINT8* y, INT32* m);
void random_crop_nhwc_cpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const UINT8* a, UINT8* y, INT32* m);
void float_to_float_convert_norm_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const float* a, float* y);
void float_to_float_convert_norm_o_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const float* a, float* y);
void scale_norm_cpu(const int N, const float scale, float* y);
void scale_norm_cpu(const int N, const float scale, const UINT8* a, float* y);
void gen_random_data(const int n, UINT32 mask, UINT32* data);
void uint8_to_uint8_grey_cpu(const int N, const UINT8* a, const Shape& shape, const int& channels, const bool has_alpha_old, float gamma, UINT8* out, const bool& cf);

//gpu
void draw_board_by_point_gpu(const int N, const int c, const int bh, const int bw, const int ch, const int cw, const UINT8* board, const UINT8* chess, const UINT32* point, const UINT8* idx, UINT8* image);
void uint8_to_uint8_scale_gpu(const int N, const UINT8* a, const Shape& shape, const int sh, const int sw, UINT8* y, const bool& cf);
void random_crop_gpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, INT32* m);
void random_crop_nhwc_gpu(const int N, const int channels, const int oh, const int ow,
    const int h, const int w, const int p, const INT32* rh, const INT32* rw, const UINT8* a, UINT8* y, INT32* m);
void horizontal_flip_gpu(const int N, const int c, const int h, const int w, UINT8* y);
void horizontal_flip_nhwc_gpu(const int N, const int c, const int h, const int w, UINT8* y);
void uint8_to_uint8_color_gpu(const int N, const UINT8* a, const Shape& shape, const float hue, const float sat, const float val, UINT8* y, const bool& cf);
void uint8_to_float_convert_norm_gpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y);
void uint8_to_float_convert_norm_o_gpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y);

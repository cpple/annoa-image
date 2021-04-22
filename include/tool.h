#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void remove_alpha_cpu(const int N, const UINT8* a, UINT8* y);
void uint8_to_float_convert_cpu(const int N, const float scale, const UINT8* a, float* y);
void uint8_to_float_convert_norm_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y);
void capture_bbox_img_cpu(const int N, const UINT8* a, UINT32 c, UINT32 aw, UINT32 ah, UINT32 x, UINT32 y, UINT32 w, UINT32 h, UINT8* r);
void uint8_to_float_random_crop_hori_norm_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y);
void uint8_to_float_random_crop_hori_norm_o_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y);

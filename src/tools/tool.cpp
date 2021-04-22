
#include "../../include/util.h"
#include "../../include/tool.h"

#include <time.h>

void remove_alpha_kernel_cpu_(const int n, const UINT8* a, UINT8* y) {
	CPU_KERNEL_LOOP(index, n) {
		int r = index * 3;
		int o = index * 4;
		y[r] = a[o];
		y[r + 1] = a[o + 1];
		y[r + 2] = a[o + 2];
	}
}

void remove_alpha_cpu(const int N, const UINT8* a, UINT8* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	remove_alpha_kernel_cpu_(N, a, y);
}

void uint8_float_convert_kernel_cpu_(const int n, const float scale, const UINT8* a, float* y) {
	CPU_KERNEL_LOOP(index, n) {
		y[index] = static_cast<float>(a[index]) * scale;
	}
}

void uint8_to_float_convert_cpu(const int N, const float scale, const UINT8* a, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)

	uint8_float_convert_kernel_cpu_(N, scale, a, y);
}

void capture_bbox_img_kernel_cpu_(const int n, const UINT8* a, UINT32 c, UINT32 aw, UINT32 ah, UINT32 x, UINT32 y, UINT32 w, UINT32 h, UINT8* r) {
	CPU_KERNEL_LOOP(index, n) {

		int spr = w * c;
		int xr = index % w;
		int yr = index / w;

		//int x1 = x - w / 2;
		//int y1 = y - h / 2;

		int cx = xr + x;
		int cy = yr + y;

		int sp = aw * c;
		for (UINT32 i = 0; i < c; i++)
		{
			int idxr = yr * spr + xr * c + i;
			int idxa = cy * sp + cx * c + i;
			r[idxr] = a[idxa];
		}   		
	}
}

void capture_bbox_img_cpu(const int N, const UINT8* a, UINT32 c, UINT32 aw, UINT32 ah, UINT32 x, UINT32 y, UINT32 w, UINT32 h, UINT8* r) {
	// NOLINT_NEXT_LINE(whitespace/operators)

	capture_bbox_img_kernel_cpu_(N, a, c, aw, ah, x, y, w, h, r);
}

void uint8_to_float_random_crop_hori_norm_kernel_cpu_(const int n, const float scale, const UINT32 b, const UINT32 c,
	const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p,
	const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y) {
	CPU_KERNEL_LOOP(index, n) {

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
		float v = 0;

		int moveH = sh - p;
		int moveW = sw - p;

		inputH = moveH + outH;
		inputW = moveW + outW;
		int spIDim = ow * oh;
		int bIDim = ow * oh * c;

		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels);
		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels]);
		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d %f %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels], s[channels]);
		if (inputH >= 0 && inputH < (int)oh && inputW >= 0 && inputW < (int)ow) {

			//printf("CUDA_KERNEL_LOOP_____:%d %d %d %d %d %d %d %d %f %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels], s[channels]);
			v = static_cast<float>(a[batch * bIDim + channels * spIDim + inputH * (int)ow + inputW]);
			v *= scale;
			if (norm) {
				v = (v - m[channels]) / s[channels];
			}
		}
		if (hori) {
			outW = (w - 1) - outW;
			index = batch * bDim + channels * spDim + outH * (int)w + outW;
		}
		y[index] = v;
	}
}

void uint8_to_float_random_crop_hori_norm_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow,
	const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw,
	const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)

	uint8_to_float_random_crop_hori_norm_kernel_cpu_(N, scale, batch, channels, oh, ow, h, w, p, sh, sw, hori, norm, m, s, a, y);
}

void uint8_to_float_random_crop_hori_norm_o_kernel_cpu_(const int n, const float scale, const UINT32 b, const UINT32 c,
	const UINT32 oh, const UINT32 ow, const UINT32 h, const UINT32 w, const UINT32 p,
	const UINT32 sh, const UINT32 sw, const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y) {
	CPU_KERNEL_LOOP(index, n) {

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
		float v = 0;

		int moveH = sh - p;
		int moveW = sw - p;

		inputH = moveH + outH;
		inputW = moveW + outW;
		//int spIDim = ow * oh;
		int bIDim = ow * oh * c;

		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels);
		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels]);
		//printf("CUDA_KERNEL_LOOP:%d %d %d %d %d %d %d %d %f %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels], s[channels]);
		if (inputH >= 0 && inputH < (int)oh && inputW >= 0 && inputW < (int)ow) {

			//printf("CUDA_KERNEL_LOOP_____:%d %d %d %d %d %d %d %d %f %f \n", inputH, inputW, moveH, moveW, outH, outW, batch, channels, m[channels], s[channels]);
			v = static_cast<float>(a[batch * bIDim + c * inputH * (int)ow + inputW * c + channels]);
			v *= scale;
			if (norm) {
				v = (v - m[channels]) / s[channels];
			}
		}
		if (hori) {
			outW = (w - 1) - outW;
			index = batch * bDim + c * outH * (int)w + outW * c + channels;
		}
		y[index] = v;
	}
}

void uint8_to_float_random_crop_hori_norm_o_cpu(const int N, const float scale, const UINT32 batch, const UINT32 channels, const UINT32 oh, const UINT32 ow,
	const UINT32 h, const UINT32 w, const UINT32 p, const UINT32 sh, const UINT32 sw,
	const bool hori, const bool norm, const float* m, const float* s, const UINT8* a, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)

	uint8_to_float_random_crop_hori_norm_o_kernel_cpu_(N, scale, batch, channels, oh, ow, h, w, p, sh, sw, hori, norm, m, s, a, y);
}

void uint8_to_float_convert_norm_kernel_cpu_(const int n, const float scale, const UINT32 b, const UINT32 c, const float* m, const float* s, const UINT8* a, float* y) {
	CPU_KERNEL_LOOP(index, n) {

		int spDim = n / (b * c);
		int bDim = spDim * c;

		int batch = index / bDim;
		int tmpBIdx = (index % bDim);
		int channels = tmpBIdx / spDim;
		float v = static_cast<float>(a[index]);
		v *= scale;
		v = (v - m[channels]) / s[channels];
		y[index] = v;
	}
}

void uint8_to_float_convert_norm_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y) {

	uint8_to_float_convert_norm_kernel_cpu_(N, scale, batch, channels, m, s, a, y);
}

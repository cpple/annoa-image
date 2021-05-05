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
        int tmpIndex = index;
		if (hori) {
            //printf(":%d \n", outW);
			outW = (w - 1) - outW;
            tmpIndex = batch * bDim + channels * spDim + outH * (int)w + outW;
            //printf("CPU_KERNEL_LOOP:%d %d %d %d %d %d %d \n", batch, bDim, channels, spDim, outH, w, outW);
		}
		y[tmpIndex] = v;
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
        int tmpIndex = index;
		if (hori) {
			outW = (w - 1) - outW;
            tmpIndex = batch * bDim + c * outH * (int)w + outW * c + channels;
		}
		y[tmpIndex] = v;
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

void uint8_to_float_convert_norm_o_kernel_cpu_(const int n, const float scale, const UINT32 b, const UINT32 c, const float* m, const float* s, const UINT8* a, float* y) {
    CPU_KERNEL_LOOP(index, n) {

        int channels = index % c;
        float v = static_cast<float>(a[index]);
        v *= scale;
        v = (v - m[channels]) / s[channels];
        y[index] = v;
    }
}

void uint8_to_float_convert_norm_o_cpu(const int N, const float scale, int batch, int channels, const float* m, const float* s, const UINT8* a, float* y) {

    uint8_to_float_convert_norm_o_kernel_cpu_(N, scale, batch, channels, m, s, a, y);
}

float get_value(
    const UINT8* data, const Shape& shape,
    const unsigned int n, const unsigned int c,
    int y, int x, const bool& cf
) {
    // Replicate border for 1 pixel

    if (x >= 0 && x < shape.w && y >= 0 && y < shape.h) {
        // N*cs*hs*ws + C*hs*ws + H*ws + W
        if (cf)
        {
            return float(data[(n * shape.h * shape.w * shape.c) +
                (y * shape.w * shape.c) + (x * shape.c) + c]);
        }
        else
        {
            return float(data[(n * shape.c * shape.h * shape.w) +
                (c * shape.h * shape.w) + (y * shape.w) + x]);
        }
    }
    else {
        return float(0);
    }
}

float cubic_interpolation(const float& d,
    const float& v1, const float& v2, const float& v3, const float& v4
) {
    // d is [0,1], marking the distance from v2 towards v3
    return v2 + d * (
        -2.0 * v1 - 3.0 * v2 + 6.0 * v3 - 1.0 * v4 + d * (
            3.0 * v1 - 6.0 * v2 + 3.0 * v3 + 0.0 * v4 + d * (
                -1.0 * v1 + 3.0 * v2 - 3.0 * v3 + 1.0 * v4))) / 6.0;
}


// Interpolate in 1D space
float interpolate_x(
    const UINT8* data, const Shape& shape,
    const unsigned int n, const unsigned int c,
    const int y, const float x, const bool& cf
) {
    float dx = x - floor(x);
    return cubic_interpolation(dx,
        get_value(data, shape, n, c, y, floor(x) - 1, cf),
        get_value(data, shape, n, c, y, floor(x), cf),
        get_value(data, shape, n, c, y, ceil(x), cf),
        get_value(data, shape, n, c, y, ceil(x) + 1, cf));
}


// Interpolate in 2D space
float interpolate_xy(
    const UINT8* data, const Shape& shape,
    const unsigned int n, const unsigned int c,
    const float y, const float x, const bool& cf
) {
    float dy = y - floor(y);
    return cubic_interpolation(dy,
        interpolate_x(data, shape, n, c, floor(y) - 1, x, cf),
        interpolate_x(data, shape, n, c, floor(y), x, cf),
        interpolate_x(data, shape, n, c, ceil(y), x, cf),
        interpolate_x(data, shape, n, c, ceil(y) + 1, x, cf));
}

void uint8_to_uint8_scale_kernel_cpu_(const int n, const UINT8* a, const Shape& shape, const float sh, const float sw, UINT8* out) {
    CPU_KERNEL_LOOP(index, n) {

        int spDim = shape.h * shape.w;
        int bDim = spDim * shape.c;

        int batch = index / bDim;
        int tmpBIdx = (index % bDim);
        int channels = tmpBIdx / spDim;
        int tmpCIdx = tmpBIdx % spDim;

        float ph = (float(shape.h) - sh) / float(2.0);
        float pw = (float(shape.w) - sw) / float(2.0);

        float y = float(tmpCIdx / shape.w) - ph;
        float x = float(tmpCIdx % shape.w) - pw;

        // scale
        if (shape.w != sw) {
            x *= float(shape.w - 1) / float(sw - 1);
        }
        if (shape.h != sh) {
            y *= float(shape.h - 1) / float(sh - 1);
        }
        //printf("%f %f %f %f %f %f \n", y, x, ph, pw, sh, sw);
        out[index] = static_cast<UINT8>(interpolate_xy(a, shape, batch, channels, y, x, false));
    }
}

void uint8_to_uint8_scale_o_kernel_cpu_(const int n, const UINT8* a, const Shape& shape, const float sh, const float sw, UINT8* out) {
    CPU_KERNEL_LOOP(index, n) {

        int bDim = shape.h * shape.w * shape.c;

        int batch = index / bDim;
        int tmpBIdx = (index % bDim);
        int channels = index % shape.c;
        int oy = tmpBIdx / (shape.w * shape.c);
        int tmpWidx = tmpBIdx % (shape.w * shape.c);
        int ox = tmpWidx / shape.c;

        float ph = (float(shape.h) - sh) / float(2.0);
        float pw = (float(shape.w) - sw) / float(2.0);

        float y = float(oy) - ph;
        float x = float(ox) - pw;

        // scale
        if (shape.w != sw) {
            x *= float(shape.w - 1) / float(sw - 1);
        }
        if (shape.h != sh) {
            y *= float(shape.h - 1) / float(sh - 1);
        }

        out[index] = static_cast<UINT8>(interpolate_xy(a, shape, batch, channels, y, x, true));
    }
}

void uint8_to_uint8_scale_cpu(const int N, const UINT8* a, const Shape& shape, const float sh, const float sw, UINT8* y, const bool& cf) {

    if (cf)
    {
        uint8_to_uint8_scale_o_kernel_cpu_(N, a, shape, sh, sw, y);
    }
    else
    {
        uint8_to_uint8_scale_kernel_cpu_(N, a, shape, sh, sw, y);
    }
}

void convert_rgb_to_hsv(float r, float g, float b, float* h, float* s, float* v) {
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

void convert_hsv_to_rgb(
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

void uint8_to_uint8_color_kernel_cpu_(const int n, const UINT8* src_data, const Shape& shape, const float hue, const float sat, const float val, UINT8* dst_data) {
    CPU_KERNEL_LOOP(index, n) {

        int spDim = shape.h * shape.w;
        int bDim = spDim * shape.c;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);

        float y = float(tmpBIdx / shape.w);
        float x = float(tmpBIdx % shape.w);

        const bool doHueRotation = (abs(hue) > FLT_EPSILON);
        const bool doDesaturation = (sat < (1.0 - 1.0 / UINT8_MAX));
        const bool doValue = (val >= 0 && val != 1);

        int bIdx = batch * bDim + y * shape.w + x;

        const int channel_stride = shape.h * shape.w;
        float norm = float(1) / float(255);
        // read
        float r = src_data[bIdx + 0 * channel_stride] * norm;
        float g = src_data[bIdx + 1 * channel_stride] * norm;
        float b = src_data[bIdx + 2 * channel_stride] * norm;

        if (doHueRotation || doDesaturation || doValue) {
            // transform
            float h, s, v;
            convert_rgb_to_hsv(r, g, b, &h, &s, &v);
            if (doHueRotation) {
                h -= hue;
            }
            if (doDesaturation) {
                s *= sat;
            }
            if (doValue) {
                v *= val;
            }
            convert_hsv_to_rgb(h, s, v, &r, &g, &b);
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
        for (int c = 3; c < shape.c; c++) {
            dst_data[bIdx + c * channel_stride] = src_data[bIdx + c * channel_stride];
        }
    }
}

void uint8_to_uint8_color_o_kernel_cpu_(const int n, const UINT8* src_data, const Shape& shape, const float hue, const float sat, const float val, UINT8* dst_data) {

    CPU_KERNEL_LOOP(index, n) {

        int spDim = shape.h * shape.w;
        int bDim = spDim * shape.c;

        int batch = index / spDim;
        int tmpBIdx = (index % spDim);

        float y = float(tmpBIdx / shape.w);
        float x = float(tmpBIdx % shape.w);

        const bool doHueRotation = (abs(hue) > FLT_EPSILON);
        const bool doDesaturation = (sat < (1.0 - 1.0 / UINT8_MAX));
        const bool doValue = (val >= 0 && val != 1);

        int bIdx = batch * bDim + y * shape.c * shape.w + x * shape.c;
        float norm = float(1) / float(255);

        // read
        float r = src_data[bIdx + 0] * norm;
        float g = src_data[bIdx + 1] * norm;
        float b = src_data[bIdx + 2] * norm;

        if (doHueRotation || doDesaturation || doValue) {
            // transform
            float h, s, v;
            convert_rgb_to_hsv(r, g, b, &h, &s, &v);
            if (doHueRotation) {
                h -= hue;
            }
            if (doDesaturation) {
                s *= sat;
            }
            if (doValue) {
                v *= val;
            }
            convert_hsv_to_rgb(h, s, v, &r, &g, &b);
        }

        UINT8 r8, g8, b8;
        r = round(r / norm);
        g = round(g / norm);
        b = round(b / norm);
        r8 = r > 255. ? UINT8(255) : (r < 0. ? UINT8(0) : static_cast<UINT8>(r));
        g8 = g > 255. ? UINT8(255) : (g < 0. ? UINT8(0) : static_cast<UINT8>(g));
        b8 = b > 255. ? UINT8(255) : (b < 0. ? UINT8(0) : static_cast<UINT8>(b));
        // write
        dst_data[bIdx + 0] = r;
        dst_data[bIdx + 1] = g;
        dst_data[bIdx + 2] = b;
        for (int c = 3; c < shape.c; c++) {
            dst_data[bIdx + c] = src_data[bIdx + c];
        }
    }
}

void uint8_to_uint8_color_cpu(const int N, const UINT8* a, const Shape& shape, const float hue, const float sat, const float val, UINT8* y, const bool& cf) {
    //https://www.cs.rit.edu/~ncs/color/t_convert.html
    if (cf)
    {
        uint8_to_uint8_color_o_kernel_cpu_(N, a, shape, hue, sat, val, y);
    }
    else
    {
        uint8_to_uint8_color_kernel_cpu_(N, a, shape, hue, sat, val, y);
    }
}

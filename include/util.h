#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <tuple>
#include <atomic>

#include <string>
#include <sstream>
#include <iostream>

typedef signed char        INT8;
typedef short              INT16;
typedef int                INT32;
typedef int                INT;
typedef long long          INT64;
typedef unsigned char      UINT8;
typedef unsigned short     UINT16;
typedef unsigned int       UINT32;
typedef std::atomic_uint   AUINT32;
typedef unsigned int       UINT;
typedef unsigned long long UINT64;
typedef float              FLOAT;
typedef double             DOUBLE;

struct Shape
{
    Shape(int _n, int _c, int _h, int _w) : n(_n), c(_c), h(_h), w(_w) {};
    Shape() {
        w = 0;
        h = 0;
        c = 0;
        n = 0;
    };
    int data_size() {
        return n * c * h * w;
    }
    int image_size() {
        return c * h * w;
    }
    int channel_size() {
        return h * w;
    }
    int grid_size() {
        return n * h * w;
    }
    const int& height() {
        return h;
    }
    const int& width() {
        return w;
    }
    const int& channel() {
        return c;
    }
    const int& number() {
        return n;
    }
    int w;
    int h;
    int c;
    int n;
};

#define CPU_KERNEL_LOOP(i, n) \
  for (int i = 0; \
       i < (n); \
       i ++)

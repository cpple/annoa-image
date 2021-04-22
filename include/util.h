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

#define CPU_KERNEL_LOOP(i, n) \
  for (int i = 0; \
       i < (n); \
       i ++)

template <typename T>
inline void checkLE(T a, T b) {
	checkIF(a >= b, "check le..");
}

template <typename T>
inline void checkGE(T a, T b) {
	checkIF(a < b, "check ge..");
}

template <typename T>
inline void checkEQ(T a, T b) {
	checkIF(a == b, "check eq..");
}

#pragma once

#ifndef _SOLVER_Stereo_UTIL_
#define _SOLVER_Stereo_UTIL_

#include "SolverUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 512 // keep consistent with the CPU

#define DR_THREAD_SIZE1_X 32
#define DR_THREAD_SIZE1_Y 8

__inline__ __device__ float warpReduce(float val) {
    int offset = 32 >> 1;
    while (offset > 0) {
        val = val + __shfl_down(val, offset, 32);
        offset = offset >> 1;
    }
    return val;
}

#endif

#pragma once

#ifndef _SOLVER_SFS_UTIL_
#define _SOLVER_SFS_UTIL_

#include "../SolverUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 1024 // keep consistent with the CPU

#define DR_THREAD_SIZE1_X 32
#define DR_THREAD_SIZE1_Y 8

extern __shared__ float bucket[];

inline __device__ void scanPart1(unsigned int threadIdx, unsigned int blockIdx, unsigned int threadsPerBlock, float* d_output)
{
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	if(threadIdx == 0) d_output[blockIdx] = bucket[0];
}

inline __device__ void scanPart2(unsigned int threadIdx, unsigned int threadsPerBlock, unsigned int blocksPerGrid, float* d_tmp)
{
	if(threadIdx < blocksPerGrid) bucket[threadIdx] = d_tmp[threadIdx];
	else						  bucket[threadIdx] = 0.0f;
	
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	__syncthreads();
}

#endif

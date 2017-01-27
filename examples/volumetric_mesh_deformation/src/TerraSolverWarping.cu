#include <iostream>


#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "CUDATimer.h"

#ifdef _WIN32
#include <conio.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define THREADS_PER_BLOCK (16*16)

#include <cutil_inline.h>
#include <cutil_math.h>

/////////////////
/// TO FLOAT3 ///
/////////////////

__global__ void reshuffleToFloat3Kernel(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height)
{
	const unsigned int N = width*height; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		d_unknown[x] = make_float3(d_x[x].x, d_x[x].y, d_a[x]);
	}
}

extern "C" void reshuffleToFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height)
{
	const unsigned int N = width * height; // Number of block variables
	reshuffleToFloat3Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_x, d_a, d_unknown, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}



///////////////////
/// FROM FLOAT3 ///
///////////////////

__global__ void reshuffleFromFloat3Kernel(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height)
{
	const unsigned int N = width*height; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		d_x[x] = make_float2(d_unknown[x].x, d_unknown[x].y);
		d_a[x] = d_unknown[x].z;
	}
}

extern "C" void reshuffleFromFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height)
{
	const unsigned int N = width * height; // Number of block variables
	reshuffleFromFloat3Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_x, d_a, d_unknown, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


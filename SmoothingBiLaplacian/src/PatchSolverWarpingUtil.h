#pragma once

#ifndef _PATCH_SOLVER_Stereo_UTIL_
#define _PATCH_SOLVER_Stereo_UTIL_

#include "SolverUtil.h"

#define PATCH_SIZE				  (16)
#define SHARED_MEM_SIZE_PATCH	  ((PATCH_SIZE+4)*(PATCH_SIZE+4))
#define SHARED_MEM_SIZE_VARIABLES ((PATCH_SIZE)*(PATCH_SIZE))

/////////////////////////////////////////////////////////////////////////
// Helper
/////////////////////////////////////////////////////////////////////////

__inline__ __device__ unsigned int getLinearThreadId(int tId_i, int tId_j)
{
	return tId_i*PATCH_SIZE+tId_j;
}

__inline__ __device__ unsigned int getLinearThreadIdCache(int tId_i, int tId_j)
{
	return (tId_i + 2)*(PATCH_SIZE + 4) + (tId_j + 2);
}

__inline__ __device__ float readValueFromCache2D(volatile float* cache, int tId_i, int tId_j)
{
	return cache[getLinearThreadIdCache(tId_i, tId_j)];
}

__inline__ __device__ float4 readValueFromCache2D(volatile float4* cache, int tId_i, int tId_j)
{
	return make_float4(cache[getLinearThreadIdCache(tId_i, tId_j)].x, cache[getLinearThreadIdCache(tId_i, tId_j)].y, cache[getLinearThreadIdCache(tId_i, tId_j)].z, cache[getLinearThreadIdCache(tId_i, tId_j)].w);
}

__inline__ __device__ bool isValid(float4& v)
{
	return v.x != MINF;
}

__inline__ __device__ void loadVariableToCache(volatile float4* cache, float4* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (isInsideImage (gId_i, gId_j, W, H)) { cache[getLinearThreadIdCache(tId_i, tId_j)].x = data[gId_i*W + gId_j].x; cache[getLinearThreadIdCache(tId_i, tId_j)].y = data[gId_i*W + gId_j].y; cache[getLinearThreadIdCache(tId_i, tId_j)].z = data[gId_i*W + gId_j].z; cache[getLinearThreadIdCache(tId_i, tId_j)].w = data[gId_i*W + gId_j].w; }
	else									{ cache[getLinearThreadIdCache(tId_i, tId_j)].x = MINF;					   cache[getLinearThreadIdCache(tId_i, tId_j)].y = MINF;				    cache[getLinearThreadIdCache(tId_i, tId_j)].z = MINF;					 cache[getLinearThreadIdCache(tId_i, tId_j)].w= MINF; }
}

__inline__ __device__ void loadPatchToCache(volatile float4* cache, float4* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, int bx, int by, int ox, int oy)
{
	unsigned int x = getLinearThreadId(tId_i, tId_j);
	for (unsigned int i = x; i < SHARED_MEM_SIZE_PATCH; i += SHARED_MEM_SIZE_VARIABLES)
	{
		int xx = i / (PATCH_SIZE + 4) - 2;
		int yy = i % (PATCH_SIZE + 4) - 2;

		const int NgId_j = bx*PATCH_SIZE + yy - ox; // global col idx
		const int NgId_i = by*PATCH_SIZE + xx - oy; // global row idx

		loadVariableToCache(cache, data, xx, yy, NgId_i, NgId_j, W, H);
	}
}

__inline__ __device__ void setZero(volatile float4* cache, int tId_i, int tId_j)
{
	cache[getLinearThreadIdCache(tId_i, tId_j)].x = 0;
	cache[getLinearThreadIdCache(tId_i, tId_j)].y = 0;
	cache[getLinearThreadIdCache(tId_i, tId_j)].z = 0;
	cache[getLinearThreadIdCache(tId_i, tId_j)].w = 0;
}

__inline__ __device__ void setPatchToZero(volatile float4* cache, int tId_i, int tId_j)
{
	unsigned int x = getLinearThreadId(tId_i, tId_j);
	for (unsigned int i = x; i < SHARED_MEM_SIZE_PATCH; i += SHARED_MEM_SIZE_VARIABLES)
	{
		int xx = i / (PATCH_SIZE + 4) - 2;
		int yy = i % (PATCH_SIZE + 4) - 2;

		setZero(cache, xx, yy);
	}
}

__inline__ __device__ void loadVariableToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	if (isInsideImage(gId_i, gId_j, W, H)) cache[getLinearThreadIdCache(tId_i, tId_j)] = data[gId_i*W + gId_j];
	else								   cache[getLinearThreadIdCache(tId_i, tId_j)] = MINF;
}

__inline__ __device__ void loadPatchToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, int bx, int by, int ox, int oy)
{
	unsigned int x = getLinearThreadId(tId_i, tId_j);
	for (unsigned int i = x; i < SHARED_MEM_SIZE_PATCH; i += SHARED_MEM_SIZE_VARIABLES)
	{
		int xx = i / (PATCH_SIZE + 4) - 2;
		int yy = i % (PATCH_SIZE + 4) - 2;

		const int NgId_j = bx*PATCH_SIZE + yy - ox; // global col idx
		const int NgId_i = by*PATCH_SIZE + xx - oy; // global row idx

		loadVariableToCache(cache, data, xx, yy, NgId_i, NgId_j, W, H);
	}
}

#endif

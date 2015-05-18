#pragma once

#ifndef _SOLVER_UTIL_
#define _SOLVER_UTIL_

#include <cutil_inline.h>
#include <cutil_math.h>
#include "cuda_SimpleMatrixUtil.h";

#define FLOAT_EPSILON 0.000001f

#ifndef BYTE
#define BYTE unsigned char
#endif

#define MINF __int_as_float(0xff800000)

///////////////////////////////////////////////////////////////////////////////
// Helper ARAP
///////////////////////////////////////////////////////////////////////////////

// Rotation Matrix
inline __device__ float3x3 evalRMat(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = CosGamma*CosBeta;
	R.m12 = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R.m13 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;

	R.m21 = SinGamma*CosBeta;
	R.m22 = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	R.m23 = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;

	R.m31 = -SinBeta;
	R.m32 = CosBeta*SinAlpha;
	R.m33 = CosBeta*CosAlpha;

	return R;
}

inline __device__ float3x3 evalRMat(const float3& angles)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	return evalRMat(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}

// Rotation Matrix
inline __device__ float3 evalRAngles(const float3x3& R)
{
	const float PI = 3.14159265359f;
	if (fabs(R.m31) != 1.0f)
	{
		const float beta = -asin(R.m31); const float cosBeta = cos(beta);
		return make_float3(atan2(R.m32 / cosBeta, R.m33 / cosBeta), beta, atan2(R.m21 / cosBeta, R.m11 / cosBeta));

		// Solution not unique, this is the second one
		//const float beta = PI+asin(R.m31); const float cosBeta = cos(beta);
		//return make_float3(atan2(R.m32/cosBeta, R.m33/cosBeta), beta, atan2(R.m21/cosBeta, R.m11/cosBeta));
	}
	else
	{
		if (R.m31 == -1.0f) return make_float3(atan2(R.m12, R.m13), PI / 2, 0.0f);
		else			   return make_float3(atan2(-R.m12, -R.m13), -PI / 2, 0.0f);
	}
}

inline __device__ float3x3 evalR(float3& angles) // angles = [alpha, beta, gamma]
{
	return evalRMat(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dAlpha
inline __device__ float3x3 evalRMat_dAlpha(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = 0.0f;
	R.m12 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R.m13 = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;

	R.m21 = 0.0f;
	R.m22 = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R.m23 = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;

	R.m31 = 0.0f;
	R.m32 = CosBeta*CosAlpha;
	R.m33 = -CosBeta*SinAlpha;

	return R;
}

inline __device__ float3x3 evalR_dAlpha(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dAlpha(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dBeta
inline __device__ float3x3 evalRMat_dBeta(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = -CosGamma*SinBeta;
	R.m12 = CosGamma*CosBeta*SinAlpha;
	R.m13 = CosGamma*CosBeta*CosAlpha;

	R.m21 = -SinGamma*SinBeta;
	R.m22 = SinGamma*CosBeta*SinAlpha;
	R.m23 = SinGamma*CosBeta*CosAlpha;

	R.m31 = -CosBeta;
	R.m32 = -SinBeta*SinAlpha;
	R.m33 = -SinBeta*CosAlpha;

	return R;
}

inline __device__ float3x3 evalR_dBeta(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dBeta(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dGamma
inline __device__ float3x3 evalRMat_dGamma(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = -SinGamma*CosBeta;
	R.m12 = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	R.m13 = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha;

	R.m21 = CosGamma*CosBeta;
	R.m22 = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R.m23 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;

	R.m31 = 0.0f;
	R.m32 = 0.0f;
	R.m33 = 0.0f;

	return R;
}

inline __device__ float3x3 evalR_dGamma(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dGamma(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dIdx
inline __device__ float3x3 evalR_dIdx(float3 angles, unsigned int idx) // 0 = alpha, 1 = beta, 2 = gamma
{
	if (idx == 0) return evalR_dAlpha(angles);
	else if (idx == 1) return evalR_dBeta(angles);
	else return evalR_dGamma(angles);
}

inline __device__ void evalDerivativeRotationMatrix(const float3& angles, float3x3& dRAlpha, float3x3& dRBeta, float3x3& dRGamma)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	dRAlpha = evalRMat_dAlpha(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRBeta = evalRMat_dBeta(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRGamma = evalRMat_dGamma(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}

///////////////////////////////////////////////////////////////////////////////
// Helper
///////////////////////////////////////////////////////////////////////////////

__inline__ __device__ void get2DIdx(int idx, unsigned int width, unsigned int height, int& i, int& j)
{
	i = idx / width;
	j = idx % width;
}

__inline__ __device__ unsigned int get1DIdx(int i, int j, unsigned int width, unsigned int height)
{
	return i*width+j;
}

__inline__ __device__ bool isInsideImage(int i, int j, unsigned int width, unsigned int height)
{
	return (i >= 0 && i < height && j >= 0 && j < width);
}

__inline__ __device__ bool inLaplacianBounds(int i, int j, unsigned int width, unsigned int height)
{
    return (i > 0 && i < (height-1) && j > 0 && j < (width - 1));
}

inline __device__ void warpReduce(volatile float* sdata, int threadIdx, unsigned int threadsPerBlock) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	if(threadIdx < 32)
	{
		if(threadIdx + 32 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx + 32];
		if(threadIdx + 16 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx + 16];
		if(threadIdx +  8 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  8];
		if(threadIdx +  4 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  4];
		if(threadIdx +  2 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  2];
		if(threadIdx +  1 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  1];
	}
}

inline __device__ void blockReduce(volatile float* sdata, int threadIdx, unsigned int threadsPerBlock)
{
	#pragma unroll
	for(unsigned int stride = threadsPerBlock/2 ; stride > 32; stride/=2)
	{
		if(threadIdx < stride) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx+stride];

		__syncthreads();
	}

	warpReduce(sdata, threadIdx, threadsPerBlock);
}

#endif

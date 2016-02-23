#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>
#include "cuda_SimpleMatrixUtil.h"

#define T_PER_BLOCK 16

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float3 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 bilinearInterpolationFloat3(float x, float y, float3* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float3 s0 = make_float3(0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float3 v00 = d_input[p00.y*imageWidth + p00.x]; if (v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float3 v10 = d_input[p10.y*imageWidth + p10.x]; if (v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float3 s1 = make_float3(0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float3 v01 = d_input[p01.y*imageWidth + p01.x]; if (v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float3 v11 = d_input[p11.y*imageWidth + p11.x]; if (v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float3 p0 = s0 / w0;
	const float3 p1 = s1 / w1;

	float3 ss = make_float3(0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		   return make_float3(MINF, MINF, MINF);
}

__global__ void resampleFloat3MapDevice(float3* d_colorMapResampledFloat4, float3* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{
		const float scaleWidth = (float)(inputWidth - 1) / (float)(outputWidth - 1);
		const float scaleHeight = (float)(inputHeight - 1) / (float)(outputHeight - 1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

		if (xInput < inputWidth && yInput < inputHeight)
		{
			d_colorMapResampledFloat4[y*outputWidth + x] = bilinearInterpolationFloat3(x*scaleWidth, y*scaleHeight, d_colorMapFloat4, inputWidth, inputHeight);
		}
	}
}

extern "C" void resampleFloat3Map(float3* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float3* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloat3MapDevice <<<blockSize, gridSize >>>(d_colorMapResampledFloat4, d_colorMapFloat4, inputWidth, inputHeight, outputWidth, outputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

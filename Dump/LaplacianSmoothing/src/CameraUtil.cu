#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"

#ifndef BYTE
#define BYTE unsigned char
#endif

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Copy Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFloatMapDevice(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = d_input[y*width+x];
}

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	copyFloatMapDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Copy Float Map Fill
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFloatMapFillDevice(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float depth = d_input[y*width+x];
	if(depth != MINF) d_output[y*width+x] = depth;
	else			  d_output[y*width+x] = 0.0f;
}

extern "C" void copyFloatMapFill(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	copyFloatMapFillDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFloat4MapDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = d_input[y*width+x];
}

extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	copyFloat4MapDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Raw Color to float
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorRawToFloatDevice(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(d_input[4*(y*width+x)+2]/255.0f, d_input[4*(y*width+x)+1]/255.0f, d_input[4*(y*width+x)+0]/255.0f, d_input[4*(y*width+x)+3]/255.0f);
}

extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertColorRawToFloatDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Color to Intensity
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorToIntensityFloatDevice(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float4 color = d_input[y*width+x];
	const float weight = 1.0f/3.0f;

	const float intensity = weight*color.x + weight*color.y + weight*color.z;

	d_output[y*width+x] = intensity;
}

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertColorToIntensityFloatDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Raw Depth to float
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthRawToFloatDevice(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = MINF;

	if(x >= width || y >= height) return;

	const unsigned short pixel = d_input[y*width+x] >> 3;
	const float depth = pixel*0.001f;

	if (depth >= minDepth && depth <= maxDepth) d_output[y*width+x] = depth;
}

extern "C" void convertDepthRawToFloat(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertDepthRawToFloatDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height, minDepth, maxDepth);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert depth map to color map view
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthToColorSpaceDevice(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInv, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const float depth = d_input[y*depthWidth+x];

	if(depth != MINF)
	{
		// Cam space depth
		float4 depthCamSpace = depthIntrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth);
		depthCamSpace = make_float4(depthCamSpace.x, depthCamSpace.y, depthCamSpace.w, 1.0f);

		// World Space
		const float4 worldSpace = depthExtrinsicsInv*depthCamSpace;
		
		// Cam space color
		float4 colorCamSpace = colorExtrinsics*worldSpace;
		colorCamSpace = make_float4(colorCamSpace.x, colorCamSpace.y, 0.0f, colorCamSpace.z);

		// Get coordinates in color image and set pixel to new depth
		const float4 screenSpaceColor  = colorIntrinsics*colorCamSpace;
		const unsigned int cx = (unsigned int)(screenSpaceColor.x/screenSpaceColor.w + 0.5f);
		const unsigned int cy = (unsigned int)(screenSpaceColor.y/screenSpaceColor.w + 0.5f);

		if(cx < colorWidth && cy < colorHeight) d_output[cy*colorWidth+cx] = screenSpaceColor.w;
	}
}

extern "C" void convertDepthToColorSpace(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInv, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const dim3 blockSize((depthWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (depthHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertDepthToColorSpaceDevice<<<blockSize, gridSize>>>(d_output, d_input, depthIntrinsicsInv, colorIntrinsics, depthExtrinsicsInv, colorExtrinsics, depthWidth, depthHeight, colorWidth, colorHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set invalid float map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setInvalidFloatMapDevice(float* d_output, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = MINF;
}

extern "C" void setInvalidFloatMap(float* d_output, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	setInvalidFloatMapDevice<<<blockSize, gridSize>>>(d_output, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set invalid float4 map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setInvalidFloat4MapDevice(float4* d_output, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF ,MINF);
}

extern "C" void setInvalidFloat4Map(float4* d_output, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	setInvalidFloat4MapDevice<<<blockSize, gridSize>>>(d_output, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Depth to Camera Space Positions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthFloatToCameraSpaceFloat4Device(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	float depth = d_input[y*width+x];

	if(depth != MINF)
	{
		float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth));
		d_output[y*width+x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.w, 1.0f);
	}
}

extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertDepthFloatToCameraSpaceFloat4Device<<<blockSize, gridSize>>>(d_output, d_input, intrinsicsInv, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral Filter Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

__global__ void bilateralFilterFloatMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width+x] = MINF;
	
	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = d_input[y*width+x];
	if(depthCenter != MINF)
	{
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = d_input[n*width+m];

					if(currentDepth != MINF)
					{
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, currentDepth-depthCenter);
								
						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}

		if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
	}
}

extern "C" void bilateralFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	bilateralFilterFloatMapDevice<<<blockSize, gridSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral Filter Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void bilateralFilterFloat4MapDevice(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width+x] = make_float4(MINF, MINF, MINF ,MINF);
	
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float4 valueCenter = d_input[y*width+x];
	if(valueCenter.x != MINF)
	{
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float4 valueCurrent = d_input[n*width+m];

					if(valueCurrent.x != MINF)
					{
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, length(valueCurrent-valueCenter));
								
						sumWeight += weight;
						sum += weight*valueCurrent;
					}
				}
			}
		}

		if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
	}
}

extern "C" void bilateralFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	bilateralFilterFloat4MapDevice<<<blockSize, gridSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeNormalsDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float4 CC = d_input[(y+0)*width+(x+0)];
		const float4 PC = d_input[(y+1)*width+(x+0)];
		const float4 CP = d_input[(y+0)*width+(x+1)];
		const float4 MC = d_input[(y-1)*width+(x+0)];
		const float4 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(PC.x, PC.y, PC.z)-make_float3(MC.x, MC.y, MC.z), make_float3(CP.x, CP.y, CP.z)-make_float3(CM.x, CM.y, CM.z));
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = make_float4(n/l, 1.0f);
			}
		}
	}
}

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeNormalsDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Derivatices Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeDerivativesFloatDevice(float* d_outputDU, float* d_outputDV, float* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_outputDU[y*width+x] = MINF;
	d_outputDV[y*width+x] = MINF;

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{												    // Stencil
		const float& p00 = d_input[(y-1)*width+(x-1)]; // 00 01 02  |  (-1,-1) (-1, 0) (-1, 1)
		const float& p01 = d_input[(y-1)*width+(x+0)]; // 10 11 12  |  ( 0,-1) ( 0, 0) ( 0, 1)
		const float& p02 = d_input[(y-1)*width+(x+1)]; // 20 21 22  |  ( 1,-1) ( 1, 0) ( 1, 1)

		const float& p10 = d_input[(y+0)*width+(x-1)]; // ------------ u ->
		const float& p12 = d_input[(y+0)*width+(x+1)]; // |
													    // |	   Image
		const float& p20 = d_input[(y+1)*width+(x-1)]; // v    Domain
		const float& p21 = d_input[(y+1)*width+(x+0)]; // |
		const float& p22 = d_input[(y+1)*width+(x+1)]; // v
		
		if(p00 == MINF || p01 == MINF || p02 == MINF || p10 == MINF || p12 == MINF || p20 == MINF|| p21 == MINF || p22 == MINF) return;

		float dU = 1.0f*(p02 - p00) + 
				   2.0f*(p12 - p10) +
				   1.0f*(p22 - p20);
		dU /= 8.0f;

		float dV = 1.0f*(p20 - p00) + //! Are these flipped ? *(-1.0f) ? Check !!!
				   2.0f*(p21 - p01) +
				   1.0f*(p22 - p02);
		dV /= 8.0f;

		d_outputDU[y*width+x] = dU;
		d_outputDV[y*width+x] = dV;
	}
}

extern "C" void computeDerivativesFloat(float* d_outputDU, float* d_outputDV, float* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeDerivativesFloatDevice<<<blockSize, gridSize>>>(d_outputDU, d_outputDV, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

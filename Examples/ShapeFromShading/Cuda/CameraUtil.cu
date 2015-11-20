#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"


#ifndef BYTE
#define BYTE unsigned char
#endif

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

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

__global__ void copyDepthFloatMapDevice(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= width || y >= height) return;

	const float depth = d_input[y*width+x];
	if (depth >= minDepth && depth <= maxDepth) 
		d_output[y*width+x] = depth;
	else
		d_output[y*width+x] = MINF;
}

extern "C" void copyDepthFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	copyDepthFloatMapDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height,minDepth, maxDepth);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Copy Float Map Fill
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initializeOptimizerMapsDevice(float* d_output, float* d_input, float* d_input2, float* d_mask, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float depth = d_input[y*width+x];
	if(d_mask[y*width+x] != MINF) { d_output[y*width+x] = depth; }
	else						  { d_output[y*width+x] = MINF; d_input[y*width+x] = MINF; d_input2[y*width+x] = MINF; }
}

extern "C" void initializeOptimizerMaps(float* d_output, float* d_input, float* d_input2, float* d_mask, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	initializeOptimizerMapsDevice<<<blockSize, gridSize>>>(d_output, d_input, d_input2, d_mask, width, height);

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
	d_output[y*width+x] = 0.299f*color.x + 0.587f*color.y + 0.114f*color.z;
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
// Convert depth map to color map view
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthToColorSpaceDevice(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInv, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < depthWidth && y < depthHeight)
	{
		const float depth = d_input[y*depthWidth+x];

		if(depth != MINF && depth < 1.0f)
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

			if(cx < colorWidth && cy < colorHeight) d_output[cy*colorWidth+cx] = screenSpaceColor.w; // Check for minimum !!!
		}
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
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

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
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

		float depth = d_input[y*width+x];

		if(depth != MINF)
		{
			float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth));
			d_output[y*width+x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.w, 1.0f);
		}
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

inline __device__ float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f-(dist*dist)/(2.0*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x)
{
	return exp(-((x*x)/(2.0f*sigma*sigma)));
}

__global__ void bilateralFilterFloatMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

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

	if(x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width+x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float4 depthCenter = d_input[y*width+x];
	for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
	{
		for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
		{		
			if(m >= 0 && n >= 0 && m < width && n < height)
			{
				const float4 currentDepth = d_input[n*width+m];
				const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, length(currentDepth-depthCenter));
			
				sum += weight*currentDepth;
				sumWeight += weight;
			}
		}
	}

	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
}

extern "C" void bilateralFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);	
	const dim3 gridSize((width + blockSize.x - 1)/blockSize.x, (height + blockSize.y - 1)/blockSize.y);	
	
	bilateralFilterFloat4MapDevice<<<gridSize,blockSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussFilterFloatMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

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

					if(currentDepth != MINF && fabs(depthCenter-currentDepth) < sigmaR)
					{
						const float weight = gaussD(sigmaD, m-x, n-y);
								
						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}
	}

	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
}

extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	gaussFilterFloatMapDevice<<<blockSize, gridSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussFilterFloat4MapDevice(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width+x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float4 depthCenter = d_input[y*width+x];
	
	for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
	{
		for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
		{		
			if(m >= 0 && n >= 0 && m < width && n < height)
			{
				const float4 currentDepth = d_input[n*width+m];

				if(length(depthCenter-currentDepth) < sigmaR)
				{
					const float weight = gaussD(sigmaD, m-x, n-y);
								
					sumWeight += weight;
					sum += weight*currentDepth;
				}
			}
		}
	}

	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
}

extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	gaussFilterFloat4MapDevice<<<blockSize, gridSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
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
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

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
			const float3 n = cross(make_float3(PC)-make_float3(MC), make_float3(CP)-make_float3(CM));
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
// Compute Normal Map 2
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeNormalsDevice2(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float4 CC = d_input[(y+0)*width+(x+0)];
		const float4 MC = d_input[(y-1)*width+(x+0)];
		const float4 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(MC)-make_float3(CC), make_float3(CM)-make_float3(CC));
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = make_float4(n/l, 1.0f);
			}
		}
	}
}

extern "C" void computeNormals2(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeNormalsDevice2<<<blockSize, gridSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Shading Value
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void evaluateLightingModelTerms(float* d_out, float4 n)
{
	d_out[0] = 1.0;
	d_out[1] = n.y;
	d_out[2] = n.z;
	d_out[3] = n.x;
	d_out[4] = n.x*n.y;
	d_out[5] = n.y*n.z;
	d_out[6] = 3*n.z*n.z - 1;
	d_out[7] = n.z*n.x;
	d_out[8] = n.x*n.x-n.y*n.y;
}

inline __device__ float evaluateLightingModel(float* d_lit, float4 n)
{
	float tmp[9];
	evaluateLightingModelTerms(tmp, n);

	float sum = 0.0f;
	for(unsigned int i = 0; i<9; i++) sum += tmp[i]*d_lit[i];

	return sum;
}

__global__ void computeShadingValueDevice(float* d_outShading, float* d_indepth, float4* d_normals, float4x4 Intrinsic, float* d_litcoeff, unsigned int width, unsigned int height)
{
	const unsigned int posx = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int posy = blockIdx.y*blockDim.y + threadIdx.y;

	if(posx >= width || posy >= height) return;

	d_outShading[posy*width+posx] = 0;

	if(posx > 0 && posx < width-1 && posy > 0 && posy < height-1)
	{
		float4 n = d_normals[posy*width+posx];

		if(n.x != MINF)
		{
			n.x = -n.x; // Change handedness
			n.z = -n.z;
			
			float shadingval = evaluateLightingModel(d_litcoeff, n);

			if(shadingval<0.0f) shadingval = 0.0f;
			if(shadingval>1.0f) shadingval = 1.0f;

			d_outShading[posy*width+posx] = shadingval;
		}
	}
}

extern "C" void computeShadingValue(float* d_outShading, float* d_indepth, float4* d_normals, float4x4 &Intrinsic, float* d_lighting, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);

	computeShadingValueDevice<<<blockSize, gridSize>>>(d_outShading, d_indepth, d_normals, Intrinsic, d_lighting, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple Segmentation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeSimpleSegmentationDevice(float* d_output, float* d_input, float depthThres, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float inputDepth = d_input[y*width+x];
	if(inputDepth != MINF && inputDepth < depthThres) d_output[y*width+x] = inputDepth;
	else											  d_output[y*width+x] = MINF;
}

extern "C" void computeSimpleSegmentation(float* d_output, float* d_input, float depthThres, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeSimpleSegmentationDevice<<<blockSize, gridSize>>>(d_output, d_input, depthThres, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Edge Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeMaskEdgeMapFloat4Device(unsigned char* d_output, float4* d_input, float* d_indepth, float threshold, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;
	
	d_output[y*width+x] = 1;
	d_output[width*height+y*width+x] = 1;

	const float thre = threshold *threshold *3.0f;
	if(x > 0 && y > 0 && x < width-1 && y < height-1)
	{	
		if(d_indepth[y*width+x] == MINF)
		{
			d_output[y*width+x] = 0;
			d_output[y*width+x-1] = 0;
			d_output[width*height+y*width+x] = 0;
			d_output[width*height+(y-1)*width+x] = 0;

			return;
		}
		
		const float4& p0 = d_input[(y+0)*width+(x+0)];
		const float4& p1 = d_input[(y+0)*width+(x+1)];
		const float4& p2 = d_input[(y+1)*width+(x+0)];

		float dU = sqrt(((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y) * (p1.y-p0.y) + (p1.z-p0.z)*(p1.z-p0.z))/3.0f);
		float dV = sqrt(((p2.x-p0.x)*(p2.x-p0.x) + (p2.y-p0.y) * (p2.y-p0.y) + (p2.z-p0.z)*(p2.z-p0.z))/3.0f);

		//float dgradx = abs(d_indepth[y*width+x-1] + d_indepth[y*width+x+1] - 2.0f * d_indepth[y*width+x]);
		//float dgrady = abs(d_indepth[y*width+x-width] + d_indepth[y*width+x+width] - 2.0f * d_indepth[y*width+x]);

		
		if(dU > thre ) d_output[y*width+x] = 0;
		if(dV > thre ) d_output[width*height+y*width+x] = 0;
	
		//remove depth discontinuities
		const int r = 1;
		const float thres = 0.01f;

		const float pCC = d_indepth[y*width+x];
		for(int i = -r; i<=r; i++)
		{
			for(int j = -r; j<=r; j++)
			{
				int currentX = x+j;
				int currentY = y+i;

				if(currentX >= 0 && currentX < width && currentY >= 0 && currentY < height)
				{
					float d = d_indepth[currentY*width+currentX];

					if(d != MINF && abs(pCC-d) > thres)
					{
						d_output[y*width+x] = 0;
						d_output[width*height+y*width+x] = 0;
						return;
					}
				}
			}
		}
	}
}

extern "C" void computeMaskEdgeMapFloat4(unsigned char* d_output, float4* d_input, float* d_indepth, float threshold, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeMaskEdgeMapFloat4Device<<<blockSize, gridSize>>>(d_output, d_input, d_indepth, threshold,width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Clear Decission Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void clearDecissionArrayPatchDepthMaskDevice(int* d_output, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) d_output[y*inputWidth+x] = 0;
}

extern "C" void clearDecissionArrayPatchDepthMask(int* d_output, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	clearDecissionArrayPatchDepthMaskDevice<<<blockSize, gridSize>>>(d_output, inputWidth, inputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Decission Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeDecissionArrayPatchDepthMaskDevice(int* d_output, float* d_input, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight)
	{
		const int patchId_x = x/patchSize;
		const int patchId_y = y/patchSize;
		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;

		const float d = d_input[y*inputWidth+x];
		if(d != MINF) atomicMax(&d_output[patchId_y*nPatchesWidth+patchId_x], 1);
	}
}

extern "C" void computeDecissionArrayPatchDepthMask(int* d_output, float* d_input, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeDecissionArrayPatchDepthMaskDevice<<<blockSize, gridSize>>>(d_output, d_input, patchSize, inputWidth, inputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Remapping Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeRemappingArrayPatchDepthMaskDevice(int* d_output, float* d_input, int* d_prefixSum, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight)
	{
		const int patchId_x = x/patchSize;
		const int patchId_y = y/patchSize;

		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;
		
		const float d = d_input[y*inputWidth+x];
		if(d != MINF) d_output[d_prefixSum[patchId_y*nPatchesWidth+patchId_x]-1] = patchId_y*nPatchesWidth+patchId_x;
	}
}

extern "C" void computeRemappingArrayPatchDepthMask(int* d_output, float* d_input, int* d_prefixSum, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeRemappingArrayPatchDepthMaskDevice<<<blockSize, gridSize>>>(d_output, d_input, d_prefixSum, patchSize, inputWidth, inputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Debug Patch Remap Array
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void DebugPatchRemapArrayDevice(float* d_mask, int* d_remapArray, unsigned int patchSize, unsigned int numElements, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x < numElements)
	{
		int patchID = d_remapArray[x];

		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;
		const int patchId_x = patchID%nPatchesWidth;
		const int patchId_y = patchID/nPatchesWidth;

		for(unsigned int i = 0; i<patchSize; i++)
		{
			for(unsigned int j = 0; j<patchSize; j++)
			{
				const int pixel_x = patchId_x*patchSize;
				const int pixel_y = patchId_y*patchSize;

				d_mask[(pixel_y+i)*inputWidth+(pixel_x+j)] = 3.0f;
			}
		}
	}
}

extern "C" void DebugPatchRemapArray(float* d_mask, int* d_remapArray, unsigned int patchSize, unsigned int numElements, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((numElements + T_PER_BLOCK*T_PER_BLOCK - 1)/(T_PER_BLOCK*T_PER_BLOCK));
	const dim3 gridSize(T_PER_BLOCK*T_PER_BLOCK);
	
	DebugPatchRemapArrayDevice<<<blockSize, gridSize>>>(d_mask, d_remapArray, patchSize, numElements, inputWidth, inputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Erode Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
__global__ void erodeDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = d_input[y*width+x];

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float oldDepth = d_input[y*width+x];
		for(int i = -structureSize; i<=structureSize; i++)
		{
			for(int j = -structureSize; j<=structureSize; j++)
			{
				if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
				{
					float depth = d_input[(y+i)*width+(x+j)];
					if(d_input[(y+i)*width+(x+j)] == MINF) // || fabs(depth-oldDepth) > 0.05f
					{
						d_output[y*width+x] = MINF;
					}
				}
			}
		}
	}
}

extern "C" void erodeDepthMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	erodeDepthMapDevice<<<blockSize, gridSize>>>(d_output, d_input, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dilateDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float sum = 0.0f;
		float count = 0.0f;
		for(int i = -structureSize; i<=structureSize; i++)
		{
			for(int j = -structureSize; j<=structureSize; j++)
			{
				if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
				{
					const float d = d_input[(y+i)*width+(x+j)];

					if(d != MINF)
					{
						sum += d;
						count += 1.0f;
					}
				}
			}
		}

		if(count > 0.0f) d_output[y*width+x] = sum/count;
		else			 d_output[y*width+x] = MINF;
	}
}

extern "C" void dilateDepthMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	dilateDepthMapDevice<<<blockSize, gridSize>>>(d_output, d_input, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float bilinearInterpolationFloat(float x, float y, float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if(v10 != MINF) { s0 +=		 alpha *v10; w0 +=		 alpha ; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if(v11 != MINF) { s1 +=		 alpha *v11; w1 +=		 alpha ;} }

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ss = 0.0f; float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return MINF;
}

__global__ void resampleFloatMapDevice(float* d_colorMapResampledFloat, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight, float* d_depthMaskMap)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < outputWidth && y < outputHeight)
	{
		const float scaleWidth  = (float)(inputWidth-1) /(float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1)/(float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth +0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight+0.5f);

		if(xInput < inputWidth && yInput < inputHeight)
		{
			bool validPrev = (d_depthMaskMap == NULL) || (d_depthMaskMap[y*outputWidth+x] != MINF);

			if(validPrev) d_colorMapResampledFloat[y*outputWidth+x] = bilinearInterpolationFloat(x*scaleWidth, y*scaleHeight, d_colorMapFloat, inputWidth, inputHeight);
			else		  d_colorMapResampledFloat[y*outputWidth+x] = MINF;
		}
	}
}

extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, float* d_depthMaskMap)
{
	const dim3 blockSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	resampleFloatMapDevice<<<blockSize, gridSize>>>(d_colorMapResampledFloat, d_colorMapFloat, inputWidth, inputHeight, outputWidth, outputHeight, d_depthMaskMap);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 bilinearInterpolationFloat4(float x, float y, float4* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != MINF) { s0 +=		alpha *v10; w0 +=		alpha ; } }

	float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF) { s1 +=		alpha *v11; w1 +=		alpha ;} }

	const float4 p0 = s0/w0;
	const float4 p1 = s1/w1;

	float4 ss = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return make_float4(MINF, MINF, MINF, MINF);
}

__global__ void resampleFloat4MapDevice(float4* d_colorMapResampledFloat4, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < outputWidth && y < outputHeight)
	{
		const float scaleWidth  = (float)(inputWidth-1) /(float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1)/(float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth +0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight+0.5f);

		if(xInput < inputWidth && yInput < inputHeight)
		{
			d_colorMapResampledFloat4[y*outputWidth+x] = bilinearInterpolationFloat4(x*scaleWidth, y*scaleHeight, d_colorMapFloat4, inputWidth, inputHeight);
		}
	}
}

extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	resampleFloat4MapDevice<<<blockSize, gridSize>>>(d_colorMapResampledFloat4, d_colorMapFloat4, inputWidth, inputHeight, outputWidth, outputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Unsigned Char Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void downsampleUnsignedCharMapDevice(unsigned char* d_MapResampled, unsigned char* d_Map, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight, unsigned int layerOffsetInput, unsigned int layerOffsetOutput)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= outputWidth || y >= outputHeight) return;

	unsigned char res = 0;

	const unsigned int inputX = 2*x;
	const unsigned int inputY = 2*y;

	if((inputY+0) < inputHeight && (inputX+0) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+0)*inputWidth + (inputX+0))];
	if((inputY+0) < inputHeight && (inputX+1) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+0)*inputWidth + (inputX+1))];
	if((inputY+1) < inputHeight && (inputX+0) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+1)*inputWidth + (inputX+0))];
	if((inputY+1) < inputHeight && (inputX+1) < inputWidth) res += d_Map[layerOffsetInput + ((inputY+1)*inputWidth + (inputX+1))];

	if(res == 4) d_MapResampled[layerOffsetOutput+(y*outputWidth+x)] = 1;
	else		 d_MapResampled[layerOffsetOutput+(y*outputWidth+x)] = 0;
}

extern "C" void downsampleUnsignedCharMap(unsigned char* d_MapResampled, unsigned int outputWidth, unsigned int outputHeight, unsigned char* d_Map, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 blockSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	downsampleUnsignedCharMapDevice<<<blockSize, gridSize>>>(d_MapResampled, d_Map, inputWidth, inputHeight, outputWidth, outputHeight, 0, 0);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	downsampleUnsignedCharMapDevice<<<blockSize, gridSize>>>(d_MapResampled, d_Map, inputWidth, inputHeight, outputWidth, outputHeight, inputWidth*inputHeight, outputWidth*outputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Edge Mask to Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertEdgeMaskToFloatDevice(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = min(d_input[y*width+x], d_input[width*height+y*width+x]);
}

extern "C" void convertEdgeMaskToFloat(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	convertEdgeMaskToFloatDevice<<<blockSize, gridSize>>>(d_output, d_input, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Erode Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void erodeDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = d_input[y*width+x];

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float oldDepth = d_input[y*width+x];
		for(int i = -structureSize; i<=structureSize; i++)
		{
			for(int j = -structureSize; j<=structureSize; j++)
			{
				if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
				{
					float depth = d_input[(y+i)*width+(x+j)];
					if(depth == MINF || depth == 0.0f || fabs(depth-oldDepth) > 0.05f)
					{
						d_output[y*width+x] = MINF;
						return;
					}
				}
			}
		}
	}
}

extern "C" void erodeDepthMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	erodeDepthMapDevice<<<blockSize, gridSize>>>(d_output, d_input, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dilateDepthMapDevice(float* d_output, float* d_input, float* d_inputOrig, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float sum = 0.0f;
		float count = 0.0f;
		float oldDepth = d_inputOrig[y*width+x];
		if(oldDepth != MINF && oldDepth != 0)
		{
			for(int i = -structureSize; i<=structureSize; i++)
			{
				for(int j = -structureSize; j<=structureSize; j++)
				{
					if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
					{
						const float d = d_input[(y+i)*width+(x+j)];

						if(d != MINF && d != 0.0f && fabs(d-oldDepth) < 0.05f)
						{
							sum += d;
							count += 1.0f;
						}
					}
				}
			}
		}

		if(count > ((2*structureSize+1)*(2*structureSize+1))/36) d_output[y*width+x] = 1.0f;
		else			 d_output[y*width+x] = MINF;
	}
}

extern "C" void dilateDepthMapMask(float* d_output, float* d_input, float* d_inputOrig, int structureSize, int width, int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	dilateDepthMapDevice<<<blockSize, gridSize>>>(d_output, d_input, d_inputOrig, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Mean Filter Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void removeDevMeanMapMaskDevice(float* d_output, float* d_input, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = d_input[y*width+x];

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float oldDepth = d_input[y*width+x];

		float mean = 0.0f;
		float meanSquared = 0.0f;
		float count = 0.0f;
		for(int i = -structureSize; i<=structureSize; i++)
		{
			for(int j = -structureSize; j<=structureSize; j++)
			{
				if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
				{
					float depth = d_input[(y+i)*width+(x+j)];
					if(depth == MINF)
					{
						depth = 8.0f;
					}

					if(depth > 0.0f)
					{
						mean		+= depth;
						meanSquared += depth*depth;
						count		+= 1.0f;
					}
				}
			}
		}

		mean/=count;
		meanSquared/=count;

		float stdDev = sqrt(meanSquared-mean*mean);

		if(fabs(oldDepth-mean) > 0.5f*stdDev)// || stdDev> 0.005f)
		{
			d_output[y*width+x] = MINF;
		}
	}
}

extern "C" void removeDevMeanMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);
	
	removeDevMeanMapMaskDevice<<<blockSize, gridSize>>>(d_output, d_input, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

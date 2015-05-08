#include <iostream>
#include <cuda_runtime.h> 
#include "CUDASolverSHLighting.h"
#include <conio.h>
#include "cutil_inline.h"
#include "cutil_math.h"

#define PIXELSKIP 3

extern __shared__ float sh_data[];

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

inline __device__ float colorToIntensity(float4 color)
{
	return 0.299f*color.x + 0.587f*color.y + 0.114f*color.z;
}

////////////////////////////////////////////////////////////////////////////////////
// Calc Lighting Matrix
////////////////////////////////////////////////////////////////////////////////////

__global__ void cal_litmat(SolverSHInput input, float4* d_normalMap, float * d_litestcashe, float thres_dep, unsigned int numLightCoeffs, unsigned int sizeOfSystem)
{
	unsigned int bias_mp = sizeOfSystem * threadIdx.x;
	for(unsigned int i=0;i<sizeOfSystem;i++) sh_data[bias_mp + i] = 0.0f;
	
	__syncthreads();

	unsigned int imgposx = PIXELSKIP*(blockIdx.x * blockDim.x  + threadIdx.x);
	unsigned int imgposy = PIXELSKIP*blockIdx.y;

	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;

	int imageind_1d =  input.width*imgposy + imgposx;
	
	if(imgposx>0 && imgposx < (input.width-1) && imgposy>0 && imgposy < (input.height-1))
	{	
		float tmpdep1 = input.d_targetDepth[imageind_1d];
				
		if(tmpdep1!=MINF && tmpdep1<thres_dep)
		{
			float tmpgrey = input.d_targetIntensity[imageind_1d];

			float4 n = d_normalMap[imageind_1d];
			if(n.x != MINF && length(n) > 0.0f)
			{
				 n.x = -n.x; // Change handedness
				 n.z = -n.z;				

				float tmp_sh_data[9]; evaluateLightingModelTerms(tmp_sh_data, n);
							

				const float ax = (ux - imgposx)/fx;
				const float ay = (uy - imgposy)/fy;
				float weight = (ax*n.x + ay*n.y - n.z)/sqrt(ax*ax+ay*ay+1.0f);
				if(weight<0.2f)
					weight = 0.0f;
				else
					weight = 1.0f;

				unsigned int linRowStart = bias_mp;
				for(unsigned int i = 0; i<numLightCoeffs; i++)
				{
					for(unsigned int j = i; j<numLightCoeffs; j++)
					{
						sh_data[linRowStart+j-i] += tmp_sh_data[i]*tmp_sh_data[j]*weight;
					}
					linRowStart += numLightCoeffs-i;
				}			
				for(unsigned int i = 0; i<numLightCoeffs; i++) sh_data[bias_mp+(sizeOfSystem-numLightCoeffs)+i] += tmpgrey*tmp_sh_data[i]*weight;
			}
		}
	}

	__syncthreads();

	unsigned int tid = threadIdx.x;
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s&&(imgposx+s)<input.width)
		{
			for(unsigned int i=0;i<sizeOfSystem;i++)
			{
				sh_data[bias_mp+i] = sh_data[bias_mp+i] + sh_data[(tid+s)*sizeOfSystem+i];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		unsigned int pos_glmm = (blockIdx.y * gridDim.x + blockIdx.x)*sizeOfSystem;
		for(unsigned int i=0;i<sizeOfSystem;i++) d_litestcashe[pos_glmm+i] = sh_data[i];
	}
}

__global__ void reduce_add_sha(float * d_intermed, const unsigned int numLenth, unsigned int numLightCoeffs, unsigned int sizeOfSystem)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

	if(myId>=numLenth)
		return;

	for(unsigned int i=0;i<sizeOfSystem;i++) sh_data[tid*sizeOfSystem+i] = d_intermed[myId*sizeOfSystem+i];
	 __syncthreads();

	// do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
		if(tid < s && (myId+s)<numLenth)
		{
			for(unsigned int i=0;i<sizeOfSystem;i++) sh_data[tid*sizeOfSystem+i] += sh_data[(tid+s)*sizeOfSystem+i];
		}
		__syncthreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{		
		for(unsigned int i=0;i<sizeOfSystem;i++) d_intermed[blockIdx.x*sizeOfSystem+i] = sh_data[i];
	}
}

extern "C" void estimateLightingSH(SolverSHInput& input, float4* d_normalMap, float *d_litestcashe, float *h_litestmat, float thres_depth)
{	
	unsigned int numLightCoeffs = 9;
	unsigned int sizeOfSystem	= 54;

	const int shmem_size = sizeof(float)*LE_THREAD_SIZE*sizeOfSystem;

	const dim3 blockSize(LE_THREAD_SIZE, 1, 1);
	const dim3 gridSize((((input.width  + PIXELSKIP - 1) / PIXELSKIP) + blockSize.x - 1) / blockSize.x, ((input.height + PIXELSKIP - 1) / PIXELSKIP), 1);
	cal_litmat<<<gridSize, blockSize, shmem_size>>>(input, d_normalMap, d_litestcashe, thres_depth, numLightCoeffs, sizeOfSystem);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
	
	unsigned int numLength = gridSize.x * gridSize.y;
	while(numLength>LE_THREAD_SIZE)
	{		
		dim3 gridSize2((numLength+blockSize.x - 1) / blockSize.x,1,1);
		reduce_add_sha<<<gridSize2, blockSize, shmem_size>>>(d_litestcashe, numLength, numLightCoeffs, sizeOfSystem);
		#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
		#endif
		numLength = gridSize2.x;
	}

	reduce_add_sha<<<1, blockSize, shmem_size>>>(d_litestcashe, numLength, numLightCoeffs, sizeOfSystem);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	cutilSafeCall(cudaMemcpy(h_litestmat,d_litestcashe, sizeOfSystem*sizeof(float), cudaMemcpyDeviceToHost));
}


////////////////////////////////////////////////////////////////////////////////////
// Reflectance/Albedo Visualization
////////////////////////////////////////////////////////////////////////////////////

__global__ void est_albedos(SolverSHInput input, float4* d_normalMap)
{	
	unsigned int posx = blockIdx.x * blockDim.x  + threadIdx.x;
	unsigned int posy = blockIdx.y * blockDim.y  + threadIdx.y;

	int imageind_1d =  input.width*posy + posx;
	
	if(posx>0 && posx < (input.width-1) && posy>0 && posy < (input.height-1))
	{	
		const float4 colorval = input.d_targetColor[imageind_1d];
		float4 n = d_normalMap[imageind_1d];

		if(n.x != MINF)
		{
			n.x = -n.x; // Change handedness
			n.z = -n.z;
			
			const float est = evaluateLightingModel(input.d_litcoeff, n);					
				
			float4 A = colorval/(est*4.0f);
			if(est>0.0f) input.d_targetAlbedo[imageind_1d] = make_float4(A.x, A.y, A.z, 1.0f);
		}
	}
}

extern "C" void estimateReflectance(SolverSHInput& input, float4* d_normalMap)
{	
	const dim3 blockSize(LE_THREAD_SIZE, LE_THREAD_SIZE, 1);
	const dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, (input.height + blockSize.y - 1) / blockSize.y, 1);
	est_albedos<<<gridSize, blockSize>>>(input, d_normalMap);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

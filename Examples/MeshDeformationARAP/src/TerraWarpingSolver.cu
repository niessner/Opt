
#include <cuda_runtime.h> 
#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 128

struct float6 {
	float array[6];
};


__global__ void convertToFloat6_Kernel(const float3* src0, const float3* src1, float6* target, unsigned int numVars) 
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numVars) {
		target[x].array[0] = src0[x].x;
		target[x].array[1] = src0[x].y;
		target[x].array[2] = src0[x].z;
		target[x].array[3] = src1[x].x;
		target[x].array[4] = src1[x].y;
		target[x].array[5] = src1[x].z;
	}
}

extern "C"  void convertToFloat6(const float3* src0, const float3* src1, float6* target, unsigned int numVars)
{
	convertToFloat6_Kernel << <(numVars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(src0, src1, target, numVars);
}



__global__ void convertFromFloat6_Kernel(const float6* source, float3* tar0, float3* tar1, unsigned int numVars)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numVars) {
		const float6& src = source[x];
		tar0[x] = make_float3(src.array[0], src.array[1], src.array[2]);
		tar1[x] = make_float3(src.array[3], src.array[4], src.array[5]);
	}
}

extern "C" void convertFromFloat6(const float6* source, float3* tar0, float3* tar1, unsigned int numVars)
{
	convertFromFloat6_Kernel << <(numVars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(source, tar0, tar1, numVars);
}
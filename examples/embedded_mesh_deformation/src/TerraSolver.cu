
#include <cuda_runtime.h> 
#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 128


struct float9 {
	float array[9];
};
struct float12 {
	float array[12];
};


__global__ void convertToFloat6_Kernel(const float3* src0, const float9* src1, float12* target, unsigned int numVars)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numVars) {
		target[x].array[0] = src0[x].x;
		target[x].array[1] = src0[x].y;
		target[x].array[2] = src0[x].z;
		for (unsigned int i = 0; i < 9; i++) {
			target[x].array[3+i] = src1[x].array[i];
		}
	}
}

extern "C"  void convertToFloat12(const float3* src0, const float9* src1, float12* target, unsigned int numVars)
{
	convertToFloat6_Kernel << <(numVars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(src0, src1, target, numVars);
}



__global__ void convertFromFloat6_Kernel(const float12* source, float3* tar0, float9* tar1, unsigned int numVars)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numVars) {
		const float12& src = source[x];
		tar0[x] = make_float3(src.array[0], src.array[1], src.array[2]);
		for (unsigned int i = 0; i < 9; i++) {
			tar1[x].array[i] = src.array[i + 3];
		}
	}
}

extern "C" void convertFromFloat12(const float12* source, float3* tar0, float9* tar1, unsigned int numVars)
{
	convertFromFloat6_Kernel << <(numVars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(source, tar0, tar1, numVars);
}
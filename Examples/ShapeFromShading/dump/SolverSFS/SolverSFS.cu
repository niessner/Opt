#include <iostream>
#include <conio.h>

#include "SolverSFSParameters.h"
#include "SolverSFSState.h"
#include "SolverSFSUtil.h"

#include "SolverSFSEquations.h"

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float d = 0.0f;
	if(x < N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];

		const float residuum = evalMinusJTFDevice(x, input, state, parameters); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		state.d_r[x] = residuum;												// store for next iteration

		const float p = state.d_precondioner[x]*residuum;						// apply preconditioner M^-1
		state.d_p[x] = p;
		
		d = residuum*p;															// x-th term of nomimator for computing alpha and denominator for computing beta
	}

	bucket[threadIdx.x] = d;

	scanPart1(threadIdx.x, blockIdx.x, blockDim.x, state.d_scanAlpha);			// sum over x-th terms to compute nominator and denominator of alpha and beta inside this block
}

__global__ void PCGInit_Kernel2(SolverInput input, SolverState state)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanAlpha);		// sum over block results to compute nominator and denominator of alpha and beta
	
	if(x < input.N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];

		state.d_rDotzOld[x] = bucket[0];									// store result for next kernel call
	}
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	unsigned int N = input.N;

	int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if(blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while(1);
	}

	Cal_Shading2depth_Grad_1d<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input,state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	PCGInit_Kernel1<<<blocksPerGrid, THREADS_PER_BLOCK, shmem_size>>>(input, state, parameters);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
		
	PCGInit_Kernel2<<<blocksPerGrid, THREADS_PER_BLOCK, shmem_size>>>(input, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N;											// Number of block variables
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	float d = 0.0f;
	if(x < N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];

		const float tmp = applyJTJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 

		state.d_Ap_X[x] = tmp;												// store for next kernel call

		d = state.d_p[x]*tmp;												// x-th term of denominator of alpha
	}

	bucket[threadIdx.x] = d;

	scanPart1(threadIdx.x, blockIdx.x, blockDim.x, state.d_scanAlpha);		// sum over x-th terms to compute denominator of alpha inside this block
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = input.N;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanAlpha);		// sum over block results to compute denominator of alpha
	const float dotProduct = bucket[0];
	
	float b = 0.0f;
	if(x < N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];

		float alpha = 0.0f;
		if(dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x]/dotProduct;  // update step size alpha
	
		state.d_delta[x] = state.d_delta[x]+alpha*state.d_p[x];				// do a decent step
		
		float r = state.d_r[x]-alpha*state.d_Ap_X[x];						// update residuum
		state.d_r[x] = r;													// store for next kernel call
	
		float z = state.d_precondioner[x]*r;								// apply preconditioner M^-1
		state.d_z[x] = z;													// save for next kernel call
		
		b = z*r;															// compute x-th term of the nominator of beta
	}

	__syncthreads();														// Only write if every thread in the block has has read bucket[0]

	bucket[threadIdx.x] = b;

	scanPart1(threadIdx.x, blockIdx.x, blockDim.x, state.d_scanBeta);		// sum over x-th terms to compute nominator of beta inside this block
}

__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int N = input.N;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanBeta);		// sum over block results to compute nominator of beta

	if(x < N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];

		const float rDotzNew = bucket[0];										// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;														 
		if(rDotzOld > FLOAT_EPSILON) beta = rDotzNew/rDotzOld;						// update step size beta
	
		state.d_rDotzOld[x] = rDotzNew;											// save new rDotz for next iteration
		state.d_p[x] = state.d_z[x]+beta*state.d_p[x];							// update decent direction
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;	// Number of block variables
		
	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if(blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while(1);
	}

	PCGStep_Kernel1<<<blocksPerGrid, THREADS_PER_BLOCK, shmem_size>>>(input, state, parameters);	
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	PCGStep_Kernel2<<<blocksPerGrid, THREADS_PER_BLOCK, shmem_size>>>(input, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	PCGStep_Kernel3<<<blocksPerGrid, THREADS_PER_BLOCK, shmem_size>>>(input, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

/////////////////////////////////////////////////////////////////////////
// InitializeMemory
/////////////////////////////////////////////////////////////////////////

void InitializeMemory(SolverInput& input, SolverState& state)
{
	const unsigned int imageDim = input.width*input.height;

	cudaMemset(state.d_p,		   0, imageDim*sizeof(float));
	cudaMemset(state.d_grad,	   0, imageDim*3*sizeof(float));
	cudaMemset(state.d_shadingdif, 0, imageDim*sizeof(float));
}

/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(x < N)
	{
		if(input.m_useRemapping) x = input.d_remapArray[x];
		state.d_x[x] = state.d_x[x] + state.d_delta[x];
	}
}

void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N; // Number of block variables
	ApplyLinearUpdateDevice<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, state, parameters);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void solveSFSStub(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	InitializeMemory(input, state);

	parameters.weightFitting = parameters.weightFittingStart;
			
	for(unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		Initialization(input, state, parameters);
		
		for(unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) PCGIteration(input, state, parameters);

		ApplyLinearUpdate(input, state, parameters);

		parameters.weightFitting += parameters.weightFittingIncrement;
	}
}

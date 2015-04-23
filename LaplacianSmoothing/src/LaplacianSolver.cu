#include <iostream>

#include "LaplacianSolverParameters.h"
#include "LaplacianSolverState.h"
#include "LaplacianSolverUtil.h"
#include "LaplacianSolverEquations.h"

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


// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		const float residuum = evalMinusJTFDevice(x, input, state, parameters); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		state.d_r[x] = residuum;										   // store for next iteration

		const float p = state.d_precondioner[x] * residuum;				   // apply preconditioner M^-1
		state.d_p[x] = p;

		d = residuum*p;													   // x-th term of nomimator for computing alpha and denominator for computing beta
	}

	bucket[threadIdx.x] = d;

	scanPart1(threadIdx.x, blockIdx.x, blockDim.x, state.d_scanAlpha);		// sum over x-th terms to compute nominator and denominator of alpha and beta inside this block
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanAlpha);		// sum over block results to compute nominator and denominator of alpha and beta

	if (x < N) state.d_rDotzOld[x] = bucket[0];								// store result for next kernel call
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(N, state);

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
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		const float tmp = applyJTJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 

		state.d_Ap_X[x] = tmp;												// store for next kernel call

		d = state.d_p[x] * tmp;												// x-th term of denominator of alpha
	}

	bucket[threadIdx.x] = d;

	scanPart1(threadIdx.x, blockIdx.x, blockDim.x, state.d_scanAlpha);		// sum over x-th terms to compute denominator of alpha inside this block
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = input.N;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanAlpha);		// sum over block results to compute denominator of alpha
	const float dotProduct = bucket[0];

	float b = 0.0f;
	if (x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;  // update step size alpha

		state.d_delta[x] = state.d_delta[x] + alpha*state.d_p[x];				// do a decent step

		float r = state.d_r[x] - alpha*state.d_Ap_X[x];						// update residuum
		state.d_r[x] = r;													// store for next kernel call

		float z = state.d_precondioner[x] * r;								// apply preconditioner M^-1
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
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	scanPart2(threadIdx.x, blockDim.x, gridDim.x, state.d_scanBeta);		// sum over block results to compute nominator of beta

	if (x < N)
	{
		const float rDotzNew = bucket[0];										// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;					// update step size beta

		state.d_rDotzOld[x] = rDotzNew;												// save new rDotz for next iteration
		state.d_p[x] = state.d_z[x] + beta*state.d_p[x];							// update decent direction
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	PCGStep_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel3 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_x[x] = state.d_x[x] + state.d_delta[x];
	}
}



void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N; // Number of block variables

	ApplyLinearUpdateDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	cutilSafeCall(cudaDeviceSynchronize());

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


/////////////////////////////////////////////////////////////////////////
// Eval Cost
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	//const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		float residual = evalFDevice(x, input, state, parameters);
        residual = warpReduce(residual);
        unsigned int laneid;
        //This command gets the lane ID within the current warp
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        if (laneid == 0) {
            atomicAdd(&state.d_sumResidual[0], residual);
        }
	}
}

__global__ void EvalAllResidualsDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		float residual = evalFDevice(x, input, state, parameters);
		state.d_r[x] = residual;
	}
}

float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	float residual = 0.0f;

	bool useGPU = true;
	if (useGPU) {
		const unsigned int N = input.N; // Number of block variables
		ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
		cutilSafeCall(cudaDeviceSynchronize());
        timer.startEvent("EvalResidual");
		EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
        timer.endEvent();
		cutilSafeCall(cudaDeviceSynchronize());

		residual = state.getSumResidual();
	}
	else {
		const unsigned int N = input.N; // Number of block variables
		EvalAllResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
		cutilSafeCall(cudaDeviceSynchronize());

		float* h_residuals = new float[input.width*input.height];
		cutilSafeCall(cudaMemcpy(h_residuals, state.d_r, sizeof(float)*input.width*input.height, cudaMemcpyDeviceToHost));

		float res = 0.0f;
		for (unsigned int i = 0; i < input.width*input.height; i++) {
			res += h_residuals[i];
		}
		delete[] h_residuals;
		residual = res;
	}


#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	return residual;
}


__global__ void GradientDecentGradientDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_delta[x] = evalMinusJTFDevice(x, input, state, parameters);
	}
}

__global__ void GradientDecentUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters, float stepSize)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_x[x] = state.d_x[x] + stepSize * state.d_delta[x]; //delta is negative
	}
}


void ComputeGradient(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
    const unsigned int N = input.N; // Number of block variables
    timer.startEvent("ComputeGradient");
    GradientDecentGradientDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
    timer.endEvent();
    cutilSafeCall(cudaDeviceSynchronize());

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

}

void UpdatePosition(SolverInput& input, SolverState& state, SolverParameters& parameters, float stepSize, CUDATimer& timer)
{
	const unsigned int N = input.N; // Number of block variables
    timer.startEvent("UpdatePosition");
	GradientDecentUpdateDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters, stepSize);
    timer.endEvent();
	cutilSafeCall(cudaDeviceSynchronize());

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}





////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void LaplacianSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    CUDATimer timer;
	printf("residual=%f\n", EvalResidual(input, state, parameters, timer));

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		Initialization(input, state, parameters);

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
			PCGIteration(input, state, parameters);
		}

		ApplyLinearUpdate(input, state, parameters);	//this should be also done in the last PCGIteration

        printf("residual=%f\n", EvalResidual(input, state, parameters, timer));

		//std::cout << "enter for next loop...\n\n" << std::endl; 
		//getchar();
	}
}

////////////////////////////////////////////////////////////////////
// Main GD Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void LaplacianSolveGDStub(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    CUDATimer timer;
	float initialLearningRate = 0.01f;
	float tolerance = 1e-10f;

	float learningLoss = 0.8f;
	float learningGain = 1.1f;
	float minLearningRate = 1e-25f;

	float learningRate = initialLearningRate;

    float residual = EvalResidual(input, state, parameters, timer);
	printf("iter=%d, residual=%f, learningRate=%f\n", 0, residual, learningRate);

	for (unsigned int i = 0; i < parameters.nNonLinearIterations; i++) {
        
		float startCost = EvalResidual(input, state, parameters, timer);
        

		float stepSize = learningRate;
        ComputeGradient(input, state, parameters, timer);

        UpdatePosition(input, state, parameters, stepSize, timer);
        
        float residual = EvalResidual(input, state, parameters, timer);
        

		//update the learningRate
		float endCost = residual;
		float maxDelta = 1.0f;
		if (endCost < startCost) {
			learningRate = learningRate * learningGain;
			if (maxDelta < tolerance) {
				//break;
			}
		}
		else {
			learningRate = learningRate * learningLoss;

			if (learningRate < minLearningRate) {
				break;
			}
		}
        timer.nextIteration();
		printf("iter=%d, residual=%f, learningRate=%f\n", i + 1, residual, learningRate);
	}
    timer.evaluate();
    

}
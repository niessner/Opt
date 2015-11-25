#include <iostream>

#include "../PatchSolverSFS/PatchSolverSFSParameters.h"
#include "../PatchSolverSFS/PatchSolverSFSState.h"
#include "SolverSFSState.h"
#include "SolverSFSUtil.h"
#include "SolverSFSEquations.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "../../ConvergenceAnalysis.h"
#include "../../CUDATimer.h"

#ifdef _WIN32
#include <conio.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define WARP_SIZE 32u
#define WARP_MASK (WARP_SIZE-1u)

/////////////////////////////////////////////////////////////////////////
// Eval Residual
////////////////////////  /////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, PatchSolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(PatchSolverInput input, SolverState state, PatchSolverParameters parameters)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int N = input.N;
    

    float residual = 0.0f;
	if (x < N)
	{
		residual = evalFDevice(x, input, state, parameters);
	}
    // Must do shuffle in entire warp
    float r = warpReduce(residual);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_sumResidual, r);
    }
}

float EvalResidual(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, CUDATimer& timer)
{
	float residual = 0.0f;

	const unsigned int N = input.N; // Number of block variables
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	cutilSafeCall(cudaDeviceSynchronize());
	timer.startEvent("EvalResidual");
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	timer.endEvent();
	cutilSafeCall(cudaDeviceSynchronize());

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

    cutilSafeCall(cudaMemcpy(&residual, &state.d_sumResidual[0], sizeof(float), cudaMemcpyDeviceToHost));

	return residual;
}

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel1(PatchSolverInput input, SolverState state, PatchSolverParameters parameters)
{
    const unsigned int N = input.N;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    float d = 0.0f;
    if (x < N)
    {
        float pre = 1.0f;
        const float residuum = evalMinusJTFDevice(x, input, state, parameters, pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
        state.d_r[x] = residuum;												 // store for next iteration
        state.d_preconditioner[x] = pre;

        const float p =  pre * residuum;					 // apply preconditioner M^-1
        state.d_p[x] = p;

        d = residuum * p;								 // x-th term of nomimator for computing alpha and denominator for computing beta
        
    }
    
    d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];								// store result for next kernel call
}

void Initialization(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

    timer.startEvent("PCGInit_Kernel1");
	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    timer.endEvent();
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	timer.startEvent("PCGInit_Kernel2");
	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(N, state);
	timer.endEvent();
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel1(PatchSolverInput input, SolverState state, PatchSolverParameters parameters)
{
	const unsigned int N = input.N;											// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		const float tmp = applyJTJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 

		state.d_Ap_X[x]  = tmp;														// store for next kernel call

		d = state.d_p[x] * tmp;													// x-th term of denominator of alpha
	}

    d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d); // sum over x-th terms to compute denominator of alpha inside this block
    }		
}

__global__ void PCGStep_Kernel2(PatchSolverInput input, SolverState state)
{
	const unsigned int N = input.N;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = bucket[0];

	float b = 0.0f;
	if (x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;  // update step size alpha

		state.d_delta[x]  = state.d_delta[x]  + alpha*state.d_p[x];				// do a decent step

		float r = state.d_r[x] - alpha*state.d_Ap_X[x];					// update residuum
		state.d_r[x] = r;													// store for next kernel call

		float z = state.d_preconditioner[x] * r;								// apply preconditioner M^-1
		state.d_z[x] = z;													// save for next kernel call

		b = z * r;														// compute x-th term of the nominator of beta
	}


    b = warpReduce(b);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanBeta, b); // sum over x-th terms to compute denominator of alpha inside this block
    }

}

__global__ void PCGStep_Kernel3(PatchSolverInput input, SolverState state)
{
	const unsigned int N = input.N;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;


	if (x < N)
	{
		const float rDotzNew = bucket[0];										// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;					// update step size beta

		state.d_rDotzOld[x] = rDotzNew;												// save new rDotz for next iteration
		state.d_p[x]  = state.d_z[x]  + beta*state.d_p[x];							// update decent direction
	}
}

void PCGIteration(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, CUDATimer& timer)
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

    timer.startEvent("PCGStep_Kernel1");
    PCGStep_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    timer.endEvent();
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	timer.startEvent("PCGStep_Kernel2");
	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	timer.endEvent();
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	timer.startEvent("PCGStep_Kernel3");
	PCGStep_Kernel3 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	timer.endEvent();
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdateDevice(PatchSolverInput input, SolverState state, PatchSolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_x[x] = state.d_x[x] + state.d_delta[x];
	}
}

void ApplyLinearUpdate(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N; // Number of block variables
    timer.startEvent("ApplyLinearUpdateDevice");
	ApplyLinearUpdateDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
    timer.endEvent();
	cutilSafeCall(cudaDeviceSynchronize()); // Hm

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void solveSFSStub(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, ConvergenceAnalysis<float>* ca)
{
    CUDATimer timer;

    timer.reset();
    parameters.weightShading = parameters.weightShadingStart;

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		float residual = EvalResidual(input, state, parameters, timer);
		printf("%i: cost: %f\n", nIter, residual);

		Initialization(input, state, parameters, timer);

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
			PCGIteration(input, state, parameters, timer);
            parameters.weightShading += parameters.weightShadingIncrement;
            if (ca != NULL) 
                ca->addSample(FunctionValue<float>(EvalResidual(input, state, parameters, timer)));
		}

		ApplyLinearUpdate(input, state, parameters, timer);	//this should be also done in the last PCGIteration

        timer.nextIteration();

	}
    timer.evaluate();


	float residual = EvalResidual(input, state, parameters, timer);
	printf("final cost: %f\n", residual);
}

__global__ void PCGStep_Kernel_SaveInitialCostJTFAndPre(PatchSolverInput input, SolverState state, PatchSolverParameters parameters,
    float* costResult, float* jtfResult, float* preResult) {

    const unsigned int N = input.N;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N)
    {
        float pre = 1.0f;
        costResult[x] = evalFDevice(x, input, state, parameters);
        
        const float residuum = evalMinusJTFDevice(x, input, state, parameters, pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
        jtfResult[x] = -2.0f*residuum;//TODO: port
        preResult[x] = pre;
    }

}

__global__ void PCGStep_Kernel_SaveJTJ(PatchSolverInput input, SolverState state, PatchSolverParameters parameters, float* jtjResult)
{
    const unsigned int N = input.N;											// Number of block variables
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N)
    {
        jtjResult[x] = applyJTJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 
    }
}


void NonPatchSaveInitialCostJTFAndPreAndJTJ(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult)
{
    const unsigned int N = input.N; // Number of block variables
    cutilSafeCall(cudaDeviceSynchronize());
    PCGStep_Kernel_SaveInitialCostJTFAndPre<< <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters, costResult, jtfResult, preResult);

    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);

    CUDATimer timer;
    timer.reset();
    Initialization(input, state, parameters, timer);
    PCGStep_Kernel_SaveJTJ<< <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters, jtjResult);

}


extern "C" void solveSFSEvalCurrentCostJTFPreAndJTJStub(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult)
{
    parameters.weightShading = parameters.weightShadingStart;


    NonPatchSaveInitialCostJTFAndPreAndJTJ(input, state, parameters, costResult, jtfResult, preResult, jtjResult);

}
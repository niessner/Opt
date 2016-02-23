#include <iostream>

#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "WarpingSolverUtil.h"
#include "WarpingSolverEquations.h"

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

/////////////////////////////////////////////////////////////////////////
// Eval Residual
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
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

float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	float residual = 0.0f;

	const unsigned int N = input.N; // Number of block variables
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	cutilSafeCall(cudaDeviceSynchronize());
	timer.startEvent("EvalResidual");
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	timer.endEvent();
	cutilSafeCall(cudaDeviceSynchronize());

	residual = state.getSumResidual();

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	return residual;
}

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
		mat3x1 aux[4]; aux[0] = mat3x1(state.d_auxFloat3CM[x]); aux[1] = mat3x1(state.d_auxFloat3CP[x]); aux[2] = mat3x1(state.d_auxFloat3MC[x]); aux[3] = mat3x1(state.d_auxFloat3PC[x]);

		const float3 residuum = evalMinusJTFDevice(x, aux, input, state, parameters); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
		state.d_r[x]  = residuum;												 // store for next iteration

		const float3 p  = state.d_precondioner[x]  * residuum;					 // apply preconditioner M^-1
		state.d_p[x] = p;

		d = dot(residuum, p);								 // x-th term of nomimator for computing alpha and denominator for computing beta
	}
	else
	{
		state.d_p[x] = make_float3(0.0f, 0.0f, 0.0f);
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

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
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

__global__ void PCGStep_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N;											// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		const float3 tmp = applyJTJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 

		state.d_Ap_X[x]  = tmp;														// store for next kernel call

		d = dot(state.d_p[x], tmp);													// x-th term of denominator of alpha
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

		state.d_delta[x]  = state.d_delta[x]  + alpha*state.d_p[x];				// do a decent step

		float3 r = state.d_r[x] - alpha*state.d_Ap_X[x];					// update residuum
		state.d_r[x] = r;													// store for next kernel call

		float3 z = state.d_precondioner[x] * r;								// apply preconditioner M^-1
		state.d_z[x] = z;													// save for next kernel call

		b = dot(z, r);														// compute x-th term of the nominator of beta
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
		state.d_p[x]  = state.d_z[x]  + beta*state.d_p[x];							// update decent direction
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
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

__global__ void ApplyLinearUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_x[x] = state.d_x[x] + state.d_delta[x];
	}
}

void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
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

/////////////////////////////////////////////////////////////////////////
// Update Aux
/////////////////////////////////////////////////////////////////////////

__global__ void UpdateAuxDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		mat3x1 aux[4]; aux[0].setZero(); aux[1].setZero(); aux[2].setZero(); aux[3].setZero();
		const float2 offsets[4] = { make_float2(0, -1), make_float2(0, 1), make_float2(-1, 0), make_float2(1, 0), };
		
		mat3x1 p = mat3x1(state.d_x[x]);
		
		for (unsigned int k = 0; k < 4; k++)
		{
			int i; int j; get2DIdx(x, input.width, input.height, i, j); const int n_i = i + offsets[k].x; const int n_j = j + offsets[k].y;
			if (isInsideImage(n_i, n_j, input.width, input.height))
			{
				mat3x1 q = mat3x1(state.d_x[get1DIdx(n_i, n_j, input.width, input.height)]);
				mat3x1 d = p - q;
				float  v = d.getTranspose()*d;
		
				if (v < parameters.weightRegularizer / parameters.weightBeta) aux[k].setZero();
				else														  aux[k] = d;
			}
		}

		state.d_auxFloat3CM[x] = aux[0]; state.d_auxFloat3CP[x] = aux[1]; state.d_auxFloat3MC[x] = aux[2]; state.d_auxFloat3PC[x] = aux[3];
	}
}

void UpdateAux(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;
	UpdateAuxDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void ImageWarpiungSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    CUDATimer timer;

	UpdateAux(input, state, parameters);

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		float residual = EvalResidual(input, state, parameters, timer);
		printf("%i: cost: %f\n", nIter, residual);

		Initialization(input, state, parameters, timer);

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
			PCGIteration(input, state, parameters, timer);
		}

		ApplyLinearUpdate(input, state, parameters, timer);	//this should be also done in the last PCGIteration

		UpdateAux(input, state, parameters);

		if (parameters.weightBeta < 1024 * 1024) parameters.weightBeta *= 2.0f;

        timer.nextIteration();
	}
    timer.evaluate();

	float residual = EvalResidual(input, state, parameters, timer);
	printf("final cost: %f\n", residual);
}

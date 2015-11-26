#include <iostream>

#include "PatchSolverSFSParameters.h"
#include "PatchSolverSFSState.h"
#include "PatchSolverSFSUtil.h"
#include "PatchSolverSFSEquations.h"
#include "../../ConvergenceAnalysis.h"
#include "../../CUDATimer.h"

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

/////////////////////////////////////////////////////////////////////////
// PCG Patch Iteration
/////////////////////////////////////////////////////////////////////////

//includes additional temporal prior constraint coming from the surface normal at previous frame
__global__ void PCGStepPatch_Kernel_SFS_BSP_Mask_Prior(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters, int ox, int oy)
{
	const unsigned int W = input.width;
	const unsigned int H = input.height;

	const int tId_j = threadIdx.x; // local col idx
	const int tId_i = threadIdx.y; // local row idx

	const int nPatchesWidth = (input.width+PATCH_SIZE-1)/PATCH_SIZE;

	int patchId = blockIdx.x;
	if(input.m_useRemapping) patchId = input.d_remapArray[patchId];

	const int patchID_x = patchId % (nPatchesWidth); //for patch shift, has to be kept consistent with the patch selection code!
	const int patchID_y = patchId / (nPatchesWidth);

	const int g_off_x = patchID_x*PATCH_SIZE-ox;
	const int g_off_y = patchID_y*PATCH_SIZE-oy;

	const int gId_j = g_off_x+threadIdx.x; // global col idx
	const int gId_i = g_off_y+threadIdx.y; // global row idx
		
	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to shared memory
	//////////////////////////////////////////////////////////////////////////////////////////

	__shared__ float		 X[SHARED_MEM_SIZE_PATCH_SFS];			loadPatchToCache_SFS(X, state.d_x, tId_i, tId_j, g_off_y, g_off_x, W, H);
	__shared__ unsigned char MaskRow[SHARED_MEM_SIZE_PATCH];
	__shared__ unsigned char MaskCol[SHARED_MEM_SIZE_PATCH];		loadMaskToCache_SFS(MaskRow, MaskCol, input.d_maskEdgeMap,tId_i, tId_j, g_off_y, g_off_x, W, H);
	__shared__ float		 P[SHARED_MEM_SIZE_PATCH_SFS];			SetPatchToZero_SFS(P, 0.0f, tId_i, tId_j, g_off_y, g_off_x);

	__syncthreads();

	__shared__ float Gradx[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Grady[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Gradz[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Shadingdif[SHARED_MEM_SIZE_PATCH_SFS_GE];		PreComputeGrad_LS(tId_j,tId_i,g_off_x, g_off_y, X, input, Gradx,Grady,Gradz, Shadingdif);
	
	__shared__ float patchBucket[SHARED_MEM_SIZE_VARIABLES];

	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to registers
	//////////////////////////////////////////////////////////////////////////////////////////
	
	register float Delta = 0.0f;
	register float R;
	register float Z;
	register float Pre;
	register float RDotZOld;
	register float AP;
	register float3 normal0;
	register float3 normal1;
	register float3 normal2;

	prior_normal_from_previous_depth(readValueFromCache2D_SFS(X, tId_i, tId_j), gId_j, gId_i, input, normal0,normal1, normal2);
	
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////
	// Initialize linear patch systems
	//////////////////////////////////////////////////////////////////////////////////////////

	float d = 0.0f;
	if(isInsideImage(gId_i, gId_j, W, H))
	{
		R = evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(tId_i, tId_j, gId_i, gId_j, W, H,Gradx,Grady,Gradz,Shadingdif, X, MaskRow, MaskCol, normal0,normal1,normal2, input, parameters, Pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 		
        R *= 2.0f; //TODO: Check if results are still okay once we fix this
		const float preRes = Pre*R;																   // apply preconditioner M^-1 
		P[getLinearShareMemLocate_SFS(tId_i, tId_j)] = preRes;									   // save for later
		d = R*preRes;
	}

	patchBucket[getLinearThreadId(tId_i, tId_j)] = d;											   // x-th term of nomimator for computing alpha and denominator for computing beta

	__syncthreads();
	blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);
	__syncthreads();

	if(isInsideImage(gId_i, gId_j, W, H)) RDotZOld = patchBucket[0];							   // read result for later on
	
	__syncthreads();
	
	//////////////////////////////////////////////////////////////////////////////////////////
	// Do patch PCG iterations
	//////////////////////////////////////////////////////////////////////////////////////////

	for(unsigned int patchIter = 0; patchIter < parameters.nPatchIterations; patchIter++)
	{
		const float currentP = P[getLinearShareMemLocate_SFS(tId_i, tId_j)];
				
		float d = 0.0f;
		if(isInsideImage(gId_i, gId_j, W, H))
		{			
			AP = applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(tId_i, tId_j, gId_i, gId_j, W, H,Gradx,Grady,Gradz,P, MaskRow, MaskCol, normal0,normal1,normal2, input, parameters);	// A x p_k  => J^T x J x p_k 
			d = currentP*AP;																																						// x-th term of denominator of alpha
		}

		patchBucket[getLinearThreadId(tId_i, tId_j)] = d;

		__syncthreads();
		blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);
		__syncthreads();
		
		const float dotProduct = patchBucket[0];

		float b = 0.0f;
		if(isInsideImage(gId_i, gId_j, W, H))
		{
			float alpha = 0.0f;
			if(dotProduct > FLOAT_EPSILON) alpha = RDotZOld/dotProduct;	    // update step size alpha
			Delta = Delta+alpha*currentP;									// do a decent step		
			R = R-alpha*AP;													// update residuum						
			Z = Pre*R;														// apply preconditioner M^-1
			b = Z*R;														// compute x-th term of the nominator of beta
		}

		__syncthreads();													// Only write if every thread in the block has has read bucket[0]

		patchBucket[getLinearThreadId(tId_i, tId_j)] = b;

		__syncthreads();
		blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);	// sum over x-th terms to compute nominator of beta inside this block
		__syncthreads();

		if(isInsideImage(gId_i, gId_j, W, H))
		{
			const float rDotzNew = patchBucket[0];												// get new nominator
			
			float beta = 0.0f;														 
			if(RDotZOld > FLOAT_EPSILON) beta = rDotzNew/RDotZOld;								// update step size beta
			RDotZOld = rDotzNew;																// save new rDotz for next iteration
			P[getLinearShareMemLocate_SFS(tId_i, tId_j)] = Z+beta*currentP;						// update decent direction
		}

		__syncthreads();
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	// Save to global memory
	//////////////////////////////////////////////////////////////////////////////////////////
	
	if(isInsideImage(gId_i, gId_j, W, H))
	{
		const int x = gId_i*W+gId_j;
		state.d_x[x] = state.d_x[x] + Delta;
	}
}


__global__ void PCGStepPatch_Kernel_SaveInitialCostJTFAndPreAndJTJ(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters, int ox, int oy, float* costResult, float* jtfResult, float* preResult, float* jtjResult)
{
    const unsigned int W = input.width;
    const unsigned int H = input.height;

    const int tId_j = threadIdx.x; // local col idx
    const int tId_i = threadIdx.y; // local row idx

    const int nPatchesWidth = (input.width + PATCH_SIZE - 1) / PATCH_SIZE;

    int patchId = blockIdx.x;
    if (input.m_useRemapping) patchId = input.d_remapArray[patchId];

    const int patchID_x = patchId % (nPatchesWidth); //for patch shift, has to be kept consistent with the patch selection code!
    const int patchID_y = patchId / (nPatchesWidth);

    const int g_off_x = patchID_x*PATCH_SIZE - ox;
    const int g_off_y = patchID_y*PATCH_SIZE - oy;

    const int gId_j = g_off_x + threadIdx.x; // global col idx
    const int gId_i = g_off_y + threadIdx.y; // global row idx

    //////////////////////////////////////////////////////////////////////////////////////////
    // CACHE data to shared memory
    //////////////////////////////////////////////////////////////////////////////////////////

    __shared__ float		 X[SHARED_MEM_SIZE_PATCH_SFS];			loadPatchToCache_SFS(X, state.d_x, tId_i, tId_j, g_off_y, g_off_x, W, H);
    __shared__ unsigned char MaskRow[SHARED_MEM_SIZE_PATCH];
    __shared__ unsigned char MaskCol[SHARED_MEM_SIZE_PATCH];		loadMaskToCache_SFS(MaskRow, MaskCol, input.d_maskEdgeMap, tId_i, tId_j, g_off_y, g_off_x, W, H);
    __shared__ float		 P[SHARED_MEM_SIZE_PATCH_SFS];			SetPatchToZero_SFS(P, 0.0f, tId_i, tId_j, g_off_y, g_off_x);

    __syncthreads();

    __shared__ float Gradx[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Grady[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Gradz[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Shadingdif[SHARED_MEM_SIZE_PATCH_SFS_GE];		PreComputeGrad_LS(tId_j, tId_i, g_off_x, g_off_y, X, input, Gradx, Grady, Gradz, Shadingdif);

    __shared__ float patchBucket[SHARED_MEM_SIZE_VARIABLES];

    //////////////////////////////////////////////////////////////////////////////////////////
    // CACHE data to registers
    //////////////////////////////////////////////////////////////////////////////////////////

    register float Delta = 0.0f;
    register float R;
    register float Z;
    register float Pre;
    register float RDotZOld;
    register float AP;
    register float3 normal0;
    register float3 normal1;
    register float3 normal2;

    prior_normal_from_previous_depth(readValueFromCache2D_SFS(X, tId_i, tId_j), gId_j, gId_i, input, normal0, normal1, normal2);

    __syncthreads();


    int resultIndex = gId_i*W + gId_j;

    //////////////////////////////////////////////////////////////////////////////////////////
    // Initialize linear patch systems
    //////////////////////////////////////////////////////////////////////////////////////////

    float d = 0.0f;
    if (isInsideImage(gId_i, gId_j, W, H))
    {
        costResult[resultIndex] = evalCost(tId_i, tId_j, gId_i, gId_j, W, H, Gradx, Grady, Gradz, Shadingdif, X, MaskRow, MaskCol, normal0, normal1, normal2, input, parameters);
        R = evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(tId_i, tId_j, gId_i, gId_j, W, H, Gradx, Grady, Gradz, Shadingdif, X, MaskRow, MaskCol, normal0, normal1, normal2, input, parameters, Pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 	
        R *= 2.0f; //TODO: port
        jtfResult[resultIndex] = -R;
        preResult[resultIndex] = Pre;
        const float preRes = Pre*R;																   // apply preconditioner M^-1 
        P[getLinearShareMemLocate_SFS(tId_i, tId_j)] = preRes;									   // save for later
        d = R*preRes;
    }

    patchBucket[getLinearThreadId(tId_i, tId_j)] = d;											   // x-th term of nomimator for computing alpha and denominator for computing beta

    __syncthreads();
    blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);
    __syncthreads();

    if (isInsideImage(gId_i, gId_j, W, H)) RDotZOld = patchBucket[0];							   // read result for later on

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Do patch PCG iterations
    //////////////////////////////////////////////////////////////////////////////////////////

    
    const float currentP = P[getLinearShareMemLocate_SFS(tId_i, tId_j)];

    if (isInsideImage(gId_i, gId_j, W, H))
    {
        AP = applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(tId_i, tId_j, gId_i, gId_j, W, H, Gradx, Grady, Gradz, P, MaskRow, MaskCol, normal0, normal1, normal2, input, parameters);	// A x p_k  => J^T x J x p_k     
        jtjResult[resultIndex] = 2.0f * AP;
    }

}


void PCGIterationPatch(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, int ox, int oy)
{
	dim3 blockSize(PATCH_SIZE, PATCH_SIZE);

	unsigned int nPatched_x = (input.width+PATCH_SIZE-1)/PATCH_SIZE; // one more for block shift!!! fix!!
	unsigned int nPatched_y = (input.height+PATCH_SIZE-1)/PATCH_SIZE; // one more for block shift!

	dim3 gridSize;
	if(input.m_useRemapping) gridSize = dim3(input.N);
	else					 gridSize = dim3(nPatched_x*nPatched_y);

	PCGStepPatch_Kernel_SFS_BSP_Mask_Prior<<<gridSize, blockSize>>>(input, state, parameters, ox, oy);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

void PCGSaveInitialCostJTFAndPreAndJTJ(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, int ox, int oy, float* costResult, float* jtfResult, float* preResult, float* jtjResult)
{
    dim3 blockSize(PATCH_SIZE, PATCH_SIZE);

    unsigned int nPatched_x = (input.width + PATCH_SIZE - 1) / PATCH_SIZE; // one more for block shift!!! fix!!
    unsigned int nPatched_y = (input.height + PATCH_SIZE - 1) / PATCH_SIZE; // one more for block shift!

    dim3 gridSize;
    if (input.m_useRemapping) gridSize = dim3(input.N);
    else					 gridSize = dim3(nPatched_x*nPatched_y);

    PCGStepPatch_Kernel_SaveInitialCostJTFAndPreAndJTJ << <gridSize, blockSize >> >(input, state, parameters, ox, oy, costResult, jtfResult, preResult, jtjResult);

    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
}



/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdatePatchDevice(unsigned int N, PatchSolverInput input, PatchSolverState state)
{
	const int nPatchesWidth = (input.width+PATCH_SIZE-1)/PATCH_SIZE;

	int patchId = blockIdx.x;
	if(input.m_useRemapping) patchId = input.d_remapArray[patchId];

	const int patchID_x = patchId % nPatchesWidth;
	const int patchID_y = patchId / nPatchesWidth;

	const int pixelID_x = patchID_x*PATCH_SIZE+threadIdx.x;
	const int pixelID_y = patchID_y*PATCH_SIZE+threadIdx.y;
	
	if(pixelID_x < input.width && pixelID_y < input.height)
	{
		int x = pixelID_y*input.width+pixelID_x;
		state.d_x[x] = state.d_x[x] + state.d_delta[x]/DEPTH_RESCALE;
	}
}

void ApplyLinearUpdatePatch(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
{
	const unsigned int N = input.N;
	dim3 blockSize(PATCH_SIZE, PATCH_SIZE);

	unsigned int nPatched_x = (input.width+PATCH_SIZE-1)/PATCH_SIZE;
	unsigned int nPatched_y = (input.height+PATCH_SIZE-1)/PATCH_SIZE;
	
	dim3 gridSize;
	if(input.m_useRemapping) gridSize = dim3(N);
	else					 gridSize = dim3(nPatched_x*nPatched_y);

	ApplyLinearUpdatePatchDevice<<<gridSize, blockSize>>>(N, input, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////
// Evaluate Residual
////////////////////////////////////////////////////////////////////

//includes additional temporal prior constraint coming from the surface normal at previous frame
__global__ void PCGStepPatch_Kernel_SFS_BSP_Mask_Prior_Residual(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters, int ox, int oy)
{
	const unsigned int W = input.width;
	const unsigned int H = input.height;

	const int tId_j = threadIdx.x; // local col idx
	const int tId_i = threadIdx.y; // local row idx

	const int nPatchesWidth = (input.width+PATCH_SIZE-1)/PATCH_SIZE;

	int patchId = blockIdx.x;
	if(input.m_useRemapping) patchId = input.d_remapArray[patchId];

	const int patchID_x = patchId % (nPatchesWidth); // +1); for patch shift, has to be kept consistent with the patch selection code!
	const int patchID_y = patchId / (nPatchesWidth);

	const int g_off_x = patchID_x*PATCH_SIZE-ox;
	const int g_off_y = patchID_y*PATCH_SIZE-oy;

	const int gId_j = g_off_x+threadIdx.x; // global col idx
	const int gId_i = g_off_y+threadIdx.y; // global row idx
		
	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to shared memory
	//////////////////////////////////////////////////////////////////////////////////////////

	__shared__ float X[SHARED_MEM_SIZE_PATCH_SFS];					loadPatchToCache_SFS(X,					state.d_x,					tId_i, tId_j, g_off_y, g_off_x, W, H);

	__shared__ unsigned char MaskRow[SHARED_MEM_SIZE_PATCH];
	__shared__ unsigned char MaskCol[SHARED_MEM_SIZE_PATCH];		loadMaskToCache_SFS(MaskRow, MaskCol, input.d_maskEdgeMap,tId_i, tId_j, g_off_y, g_off_x, W, H);
	
	__syncthreads();

	__shared__ float Gradx[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Grady[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Gradz[SHARED_MEM_SIZE_PATCH_SFS_GE];
	__shared__ float Shadingdif[SHARED_MEM_SIZE_PATCH_SFS_GE];		PreComputeGrad_LS(tId_j,tId_i,g_off_x, g_off_y, X, input, Gradx,Grady,Gradz, Shadingdif);

	__shared__ float P[SHARED_MEM_SIZE_PATCH_SFS];					SetPatchToZero_SFS(P,	0.0f,	tId_i, tId_j, g_off_y, g_off_x);

	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to registers
	//////////////////////////////////////////////////////////////////////////////////////////
	
	register float Pre;
	register float3 normal0;
	register float3 normal1;
	register float3 normal2;

	prior_normal_from_previous_depth(readValueFromCache2D_SFS(X, tId_i, tId_j), gId_j, gId_i, input, normal0, normal1, normal2);
	
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////
	// Initialize linear patch systems
	//////////////////////////////////////////////////////////////////////////////////////////

	if(isInsideImage(gId_i, gId_j, W, H))
	{
		float R = evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(tId_i, tId_j, gId_i, gId_j, W, H,Gradx,Grady,Gradz,Shadingdif, X, MaskRow, MaskCol, normal0,normal1,normal2, input, parameters, Pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
		atomicAdd(&state.d_residual[0], abs(R));
		atomicAdd(&state.d_residual[1], 1.0f);
	}
}

float evaluateResidual(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
{
	float residual = 0.0f;
	float count = 0.0f;
	cutilSafeCall(cudaMemcpy(&state.d_residual[0], &residual, sizeof(float), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&state.d_residual[1], &count, sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockSize(PATCH_SIZE, PATCH_SIZE);

	unsigned int nPatched_x = (input.width+PATCH_SIZE-1)/PATCH_SIZE;
	unsigned int nPatched_y = (input.height+PATCH_SIZE-1)/PATCH_SIZE;

	dim3 gridSize;
	if(input.m_useRemapping) gridSize = dim3(input.N);
	else					 gridSize = dim3(nPatched_x*nPatched_y);

	PCGStepPatch_Kernel_SFS_BSP_Mask_Prior_Residual<<<gridSize, blockSize>>>(input, state, parameters, 0, 0);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	cutilSafeCall(cudaMemcpy(&residual, &state.d_residual[0], sizeof(float), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&count,	&state.d_residual[1], sizeof(float), cudaMemcpyDeviceToHost));

	return residual/count;
}


__global__ void PCGStepPatch_Kernel_SFS_BSP_Mask_Prior_Cost(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters, int ox, int oy)
{
    const unsigned int W = input.width;
    const unsigned int H = input.height;

    const int tId_j = threadIdx.x; // local col idx
    const int tId_i = threadIdx.y; // local row idx

    const int nPatchesWidth = (input.width + PATCH_SIZE - 1) / PATCH_SIZE;

    int patchId = blockIdx.x;
    if (input.m_useRemapping) patchId = input.d_remapArray[patchId];

    const int patchID_x = patchId % (nPatchesWidth); // +1); for patch shift, has to be kept consistent with the patch selection code!
    const int patchID_y = patchId / (nPatchesWidth);

    const int g_off_x = patchID_x*PATCH_SIZE - ox;
    const int g_off_y = patchID_y*PATCH_SIZE - oy;

    const int gId_j = g_off_x + threadIdx.x; // global col idx
    const int gId_i = g_off_y + threadIdx.y; // global row idx

    //////////////////////////////////////////////////////////////////////////////////////////
    // CACHE data to shared memory
    //////////////////////////////////////////////////////////////////////////////////////////

    __shared__ float X[SHARED_MEM_SIZE_PATCH_SFS];					loadPatchToCache_SFS(X, state.d_x, tId_i, tId_j, g_off_y, g_off_x, W, H);

    __shared__ unsigned char MaskRow[SHARED_MEM_SIZE_PATCH];
    __shared__ unsigned char MaskCol[SHARED_MEM_SIZE_PATCH];		loadMaskToCache_SFS(MaskRow, MaskCol, input.d_maskEdgeMap, tId_i, tId_j, g_off_y, g_off_x, W, H);

    __syncthreads();

    __shared__ float Gradx[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Grady[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Gradz[SHARED_MEM_SIZE_PATCH_SFS_GE];
    __shared__ float Shadingdif[SHARED_MEM_SIZE_PATCH_SFS_GE];		PreComputeGrad_LS(tId_j, tId_i, g_off_x, g_off_y, X, input, Gradx, Grady, Gradz, Shadingdif);


    //////////////////////////////////////////////////////////////////////////////////////////
    // CACHE data to registers
    //////////////////////////////////////////////////////////////////////////////////////////

    register float Pre;
    register float3 normal0;
    register float3 normal1;
    register float3 normal2;

    prior_normal_from_previous_depth(readValueFromCache2D_SFS(X, tId_i, tId_j), gId_j, gId_i, input, normal0, normal1, normal2);

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////
    // Initialize linear patch systems
    //////////////////////////////////////////////////////////////////////////////////////////

    if (isInsideImage(gId_i, gId_j, W, H))
    {
        float cost = evalCost(tId_i, tId_j, gId_i, gId_j, W, H, Gradx, Grady, Gradz, Shadingdif, X, MaskRow, MaskCol, normal0, normal1, normal2, input, parameters);
        atomicAdd(&state.d_residual[0], cost);
    }
}

float evaluateCost(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
{
    float cost = 0.0f;
    cutilSafeCall(cudaMemcpy(&state.d_residual[0], &cost, sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(PATCH_SIZE, PATCH_SIZE);

    unsigned int nPatched_x = (input.width + PATCH_SIZE - 1) / PATCH_SIZE;
    unsigned int nPatched_y = (input.height + PATCH_SIZE - 1) / PATCH_SIZE;

    dim3 gridSize;
    if (input.m_useRemapping) gridSize = dim3(input.N);
    else					 gridSize = dim3(nPatched_x*nPatched_y);

    PCGStepPatch_Kernel_SFS_BSP_Mask_Prior_Cost << <gridSize, blockSize >> >(input, state, parameters, 0, 0);
#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

    cutilSafeCall(cudaMemcpy(&cost, &state.d_residual[0], sizeof(float), cudaMemcpyDeviceToHost));

    return cost;
}


////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

int offsetX[8] = {(int)(0.0f*PATCH_SIZE), (int)((1.0f/2.0f)*PATCH_SIZE), (int)((1.0f/4.0f)*PATCH_SIZE), (int)((3.0f/4.0f)*PATCH_SIZE), (int)((1.0f/8.0f)*PATCH_SIZE), (int)((5.0f/8.0f)*PATCH_SIZE), (int)((3.0f/8.0f)*PATCH_SIZE), (int)((7.0f/8.0f)*PATCH_SIZE)}; // Halton sequence base 2
int offsetY[8] = {(int)(0.0f*PATCH_SIZE), (int)((1.0f/3.0f)*PATCH_SIZE), (int)((2.0f/3.0f)*PATCH_SIZE), (int)((1.0f/9.0f)*PATCH_SIZE), (int)((4.0f/9.0f)*PATCH_SIZE), (int)((7.0f/9.0f)*PATCH_SIZE), (int)((2.0f/9.0f)*PATCH_SIZE), (int)((5.0f/9.0f)*PATCH_SIZE)}; // Halton sequence base 3

extern "C" void patchSolveSFSStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, ConvergenceAnalysis<float>* ca)
{
    CUDATimer t;
    t.reset();
	parameters.weightShading = parameters.weightShadingStart;

	unsigned int nPatched_x = (input.width+PATCH_SIZE-1)/PATCH_SIZE;
	unsigned int nPatched_y = (input.height+PATCH_SIZE-1)/PATCH_SIZE;
	//if(input.m_useRemapping) std::cout << "Number of Variables: " << input.N*PATCH_SIZE*PATCH_SIZE << std::endl;
	//else					 std::cout << "Number of Variables: " << nPatched_x*nPatched_y*PATCH_SIZE*PATCH_SIZE << std::endl;
    
	for(unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{	

		int o = 0;
		for(unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
		{
            t.startEvent("evaluateCost");
            float cost = evaluateCost(input, state, parameters);
            t.endEvent();
            printf("iteration (%d,%d) cost=%f\n", nIter, linIter, cost);
            t.startEvent("PCGIterationPatch");
            PCGIterationPatch(input, state, parameters, offsetX[o], offsetY[o]);
            t.endEvent();
			o = (o+1)%8;
			parameters.weightShading += parameters.weightShadingIncrement;

			if(ca != NULL) ca->addSample(FunctionValue<float>(evaluateResidual(input, state, parameters)));
		}
	}
    t.startEvent("evaluateCost");
    float finalCost = evaluateCost(input, state, parameters);
    printf("Final Cost: %f\n",  finalCost);
    t.endEvent();
    t.evaluate();
}

extern "C" void patchSolveSFSEvalCurrentCostJTFPreAndJTJStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult)
{
    parameters.weightShading = parameters.weightShadingStart;

    unsigned int nPatched_x = (input.width + PATCH_SIZE - 1) / PATCH_SIZE;
    unsigned int nPatched_y = (input.height + PATCH_SIZE - 1) / PATCH_SIZE;

    int o = 0;
    PCGSaveInitialCostJTFAndPreAndJTJ(input, state, parameters, offsetX[o], offsetY[o], costResult, jtfResult, preResult, jtjResult);
    
}

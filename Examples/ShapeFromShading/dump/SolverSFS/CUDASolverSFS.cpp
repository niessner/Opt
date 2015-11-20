#include "stdafx.h"

#include "CUDASolverSFS.h"

extern "C" void copyFloatMapFill(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void solveSFSStub(SolverInput& input, SolverState& state, SolverParameters& parameters);

extern "C" void computeDecissionArrayDepthMask(int* d_output, float* d_input, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void computeRemappingArrayDepthMask(int* d_output, float* d_input, int* d_prefixSum, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void DebugRemapArray(float* d_mask, int* d_remapArray, unsigned int numElements);

CUDASolverSFS::CUDASolverSFS(const Matrix4f& intrinsics, unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight), ICUDASolverSFS(intrinsics)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

	const unsigned int N = m_imageWidth*m_imageHeight;
	const unsigned int numberOfVariables = N;
	
	const unsigned int numberOfResiduums = (m_imageHeight-2)*(m_imageWidth-2)*3+N;

	cutilSafeCall(cudaMalloc(&d_decissionArrayDepthMask, sizeof(int)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&d_prefixSumDepthMask, sizeof(int)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&d_remappingArrayDepthMask, sizeof(int)*numberOfVariables));

	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_grad,			sizeof(float)*3*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_shadingdif,	sizeof(float)*numberOfVariables));

	m_solverInput.N = m_imageWidth*m_imageHeight;
	m_solverInput.width = m_imageWidth;
	m_solverInput.height = m_imageHeight;
	m_solverInput.d_remapArray = NULL;

	m_solverInput.calibparams.fx =  m_intrinsics(0, 0);
	m_solverInput.calibparams.fy = -m_intrinsics(1, 1);
	m_solverInput.calibparams.ux =  m_intrinsics(0, 3);
	m_solverInput.calibparams.uy =  m_intrinsics(1, 3);
}

CUDASolverSFS::~CUDASolverSFS()
{
	cutilSafeCall(cudaFree(d_decissionArrayDepthMask));
	cutilSafeCall(cudaFree(d_prefixSumDepthMask));
	cutilSafeCall(cudaFree(d_remappingArrayDepthMask));

	// State
	cutilSafeCall(cudaFree(m_solverState.d_delta));
	cutilSafeCall(cudaFree(m_solverState.d_r));
	cutilSafeCall(cudaFree(m_solverState.d_z));
	cutilSafeCall(cudaFree(m_solverState.d_p));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_X));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondioner));
	cutilSafeCall(cudaFree(m_solverState.d_grad));
	cutilSafeCall(cudaFree(m_solverState.d_shadingdif));
}

void CUDASolverSFS::solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, float* d_litcoeff, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightFittingIncrement, float weightShading, float weightBoundary, float weightRegularizer, bool useRemapping, float* d_outputDepth)
{
	unsigned int numElements = 0;
	if(useRemapping)
	{
		computeDecissionArrayDepthMask(d_decissionArrayDepthMask, d_depthMapMaskFloat, m_solverInput.width, m_solverInput.height);
		numElements = m_scan.prefixSum(m_solverInput.width*m_solverInput.height, d_decissionArrayDepthMask, d_prefixSumDepthMask);
		computeRemappingArrayDepthMask(d_remappingArrayDepthMask, d_depthMapMaskFloat, d_prefixSumDepthMask, m_solverInput.width, m_solverInput.height);
	}

	m_solverState.d_x = d_outputDepth;
	
	m_solverInput.d_targetIntensity = d_targetIntensity;
	m_solverInput.d_targetDepth = d_targetDepth;
	m_solverInput.d_depthMapRefinedLastFrameFloat = d_depthMapRefinedLastFrameFloat;
	m_solverInput.d_litcoeff = d_litcoeff;
	m_solverInput.d_remapArray = d_remappingArrayDepthMask;
	if(useRemapping) m_solverInput.N = numElements;
	m_solverInput.m_useRemapping = useRemapping;

	SolverParameters parameters;
	parameters.weightFittingStart = weightFittingStart;
	parameters.weightFittingIncrement = weightFittingIncrement;
	parameters.weightRegularizer = weightRegularizer;
	parameters.weightShading = weightShading;
	parameters.weightBoundary = weightBoundary;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;

	solveSFSStub(m_solverInput, m_solverState, parameters);
}

void CUDASolverSFS::solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, float* d_litcoeff, float4* d_albedo, unsigned char* d_maskEdgeMap, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightFittingIncrement, float weightShading, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* d_outputDepth)
{
	unsigned int numElements = 0;
	if(useRemapping)
	{
		computeDecissionArrayDepthMask(d_decissionArrayDepthMask, d_depthMapMaskFloat, m_solverInput.width, m_solverInput.height);
		numElements = m_scan.prefixSum(m_solverInput.width*m_solverInput.height, d_decissionArrayDepthMask, d_prefixSumDepthMask);
		computeRemappingArrayDepthMask(d_remappingArrayDepthMask, d_depthMapMaskFloat, d_prefixSumDepthMask, m_solverInput.width, m_solverInput.height);
	}

	m_solverState.d_x = d_outputDepth;
	
	m_solverInput.d_targetIntensity = d_targetIntensity;
	m_solverInput.d_targetDepth = d_targetDepth;
	m_solverInput.d_depthMapRefinedLastFrameFloat = d_depthMapRefinedLastFrameFloat;
	m_solverInput.d_litcoeff = d_litcoeff;
	m_solverInput.d_remapArray = d_remappingArrayDepthMask;
	if(useRemapping) m_solverInput.N = numElements;
	m_solverInput.m_useRemapping = useRemapping;

	SolverParameters parameters;
	parameters.weightFittingStart = weightFittingStart;
	parameters.weightFittingIncrement = weightFittingIncrement;
	parameters.weightRegularizer = weightRegularizer;
	parameters.weightShading = weightShading;
	parameters.weightBoundary = weightBoundary;
	parameters.weightPrior = weightPrior;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;


	parameters.weightFittingStart = 0.0f;
	parameters.weightRegularizer = 300.0f;
	parameters.weightShading = 10.0f;

	solveSFSStub(m_solverInput, m_solverState, parameters);
}


void CUDASolverSFS::transferDetailSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat,Matrix4f rigidM, float* d_depthMapMaskFloat, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* outputDepth)
{


}

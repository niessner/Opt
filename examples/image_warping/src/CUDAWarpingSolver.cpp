#include "CUDAWarpingSolver.h"

extern "C" void ImageWarpiungSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters);	// gauss newton

CUDAWarpingSolver::CUDAWarpingSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

	const unsigned int N = m_imageWidth*m_imageHeight;
	const unsigned int numberOfVariables = N;

	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_deltaA,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rA,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_zA,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_pA,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_XA,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float2)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondionerA,sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));
}

CUDAWarpingSolver::~CUDAWarpingSolver()
{
	// State
	cutilSafeCall(cudaFree(m_solverState.d_delta));
	cutilSafeCall(cudaFree(m_solverState.d_deltaA));
	cutilSafeCall(cudaFree(m_solverState.d_r));
	cutilSafeCall(cudaFree(m_solverState.d_rA));
	cutilSafeCall(cudaFree(m_solverState.d_z));
	cutilSafeCall(cudaFree(m_solverState.d_zA));
	cutilSafeCall(cudaFree(m_solverState.d_p));
	cutilSafeCall(cudaFree(m_solverState.d_pA));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_X));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_XA));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondioner));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerA));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
}

void CUDAWarpingSolver::solveGN(float2* d_urshape, float2* d_warpField, float* d_warpAngles, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer)
{
	m_solverState.d_urshape = d_urshape;
	m_solverState.d_mask = d_mask;
	m_solverState.d_x = d_warpField;
	m_solverState.d_A = d_warpAngles;

	SolverParameters parameters;
	parameters.weightFitting = weightFitting;
	parameters.weightRegularizer = weightRegularizer;	
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;
	
	SolverInput solverInput;
	solverInput.N = m_imageWidth*m_imageHeight;
	solverInput.width = m_imageWidth;
	solverInput.height = m_imageHeight;
	solverInput.d_constraints = d_constraints;

	ImageWarpiungSolveGNStub(solverInput, m_solverState, parameters);
}

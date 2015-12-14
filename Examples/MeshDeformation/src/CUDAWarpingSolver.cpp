#include "CUDAWarpingSolver.h"

extern "C" void ImageWarpiungSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters);	// gauss newton

CUDAWarpingSolver::CUDAWarpingSolver(unsigned int N) : m_N(N)
{
	const unsigned int THREADS_PER_BLOCK = 512; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;
	const unsigned int numberOfVariables = N;

	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_deltaA,		sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rA,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_zA,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_pA,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_A,			sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondionerA, sizeof(float3)*numberOfVariables));
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
	cutilSafeCall(cudaFree(m_solverState.d_Ap_A));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondioner));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerA));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
}

void CUDAWarpingSolver::solveGN(float3* d_vertexPosFloat3, float3* d_anglesFloat3, float3* d_vertexPosFloat3Urshape, int* d_numNeighbours, int* d_neighbourIdx, int* d_neighbourOffset, float3* d_vertexPosTargetFloat3, int nonLinearIter, int linearIter, float weightFit, float weightReg)
{
	m_solverState.d_urshape = d_vertexPosFloat3Urshape;
	m_solverState.d_a = d_anglesFloat3;
	m_solverState.d_target = d_vertexPosTargetFloat3;
	m_solverState.d_x = d_vertexPosFloat3;
	
	SolverParameters parameters;
	parameters.weightFitting = weightFit;
	parameters.weightRegularizer = weightReg;
	parameters.nNonLinearIterations = nonLinearIter;
	parameters.nLinIterations = linearIter;
	
	SolverInput solverInput;
	solverInput.N = m_N;
	solverInput.d_numNeighbours = d_numNeighbours;
	solverInput.d_neighbourIdx = d_neighbourIdx;
	solverInput.d_neighbourOffset = d_neighbourOffset;

	ImageWarpiungSolveGNStub(solverInput, m_solverState, parameters);
}

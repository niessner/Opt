#include "CUDALaplacianSolver.h"

extern "C" void copyFloatMapFill(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void solveStereoStub(SolverInput& input, SolverState& state, SolverParameters& parameters);
extern "C" void estimateLightingSH(SolverInput& input, SolverState& state);

CUDALaplacianSolver::CUDALaplacianSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

	const unsigned int N = m_imageWidth*m_imageHeight;
	const unsigned int numberOfVariables = N;

	const unsigned int numberOfResiduums = N+4*N;

	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float)*numberOfVariables));
	//cutilSafeCall(cudaMalloc(&m_solverState.d_Jp,			sizeof(float)*numberOfResiduums));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)*tmpBufferSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));

}

CUDALaplacianSolver::~CUDALaplacianSolver()
{
	// State
	cutilSafeCall(cudaFree(m_solverState.d_delta));
	cutilSafeCall(cudaFree(m_solverState.d_r));
	cutilSafeCall(cudaFree(m_solverState.d_z));
	cutilSafeCall(cudaFree(m_solverState.d_p));
	//cutilSafeCall(cudaFree(m_solverState.d_Jp));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_X));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondioner));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
}


void CUDALaplacianSolver::solve(float* d_targetDepth, float* d_result, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer)
{
	//copyFloatMapFill(d_result, d_targetDepth, m_imageWidth, m_imageHeight);

	m_solverState.d_x = d_result;

	SolverParameters parameters;
	parameters.weightFitting = weightFitting;
	parameters.weightRegularizer = weightRegularizer;	
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;
	
	SolverInput solverInput;
	solverInput.N = m_imageWidth*m_imageHeight;
	solverInput.width = m_imageWidth;
	solverInput.height = m_imageHeight;
	solverInput.d_targetDepth = d_targetDepth;


	solveStereoStub(solverInput, m_solverState, parameters);
}

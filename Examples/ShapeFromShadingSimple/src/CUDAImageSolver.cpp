#include "CUDAImageSolver.h"
#include "ConvergenceAnalysis.h"


extern "C" void solveSFSStub(SolverInput& input, SolverState& state, SolverParameters& parameters, ConvergenceAnalysis<float>* ca);
extern "C" void solveSFSEvalCurrentCostJTFPreAndJTJStub(SolverInput& input, SolverState& state, SolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult);


CUDAImageSolver::CUDAImageSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

	const unsigned int N = m_imageWidth*m_imageHeight;
    const size_t unknownStorageSize = sizeof(float)*N;


	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_delta,		unknownStorageSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_r,            unknownStorageSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_z,            unknownStorageSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_p,            unknownStorageSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X,         unknownStorageSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*N));
    cutilSafeCall(cudaMalloc(&m_solverState.d_preconditioner, unknownStorageSize));
	cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));

    // Solver-specific intermediates
    cutilSafeCall(cudaMalloc(&m_solverState.B_I    , sizeof(float)*N));
    cutilSafeCall(cudaMalloc(&m_solverState.B_I_dx0, sizeof(float)*N));
    cutilSafeCall(cudaMalloc(&m_solverState.B_I_dx1, sizeof(float)*N));
    cutilSafeCall(cudaMalloc(&m_solverState.B_I_dx2, sizeof(float)*N));
}

CUDAImageSolver::~CUDAImageSolver()
{
	// State
	cutilSafeCall(cudaFree(m_solverState.d_delta));
	cutilSafeCall(cudaFree(m_solverState.d_r));
	cutilSafeCall(cudaFree(m_solverState.d_z));
	cutilSafeCall(cudaFree(m_solverState.d_p));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_X));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
    cutilSafeCall(cudaFree(m_solverState.d_preconditioner));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));

    // Solver-specific intermediates
    cutilSafeCall(cudaFree(m_solverState.B_I    ));
    cutilSafeCall(cudaFree(m_solverState.B_I_dx0));
    cutilSafeCall(cudaFree(m_solverState.B_I_dx1));
    cutilSafeCall(cudaFree(m_solverState.B_I_dx2));
}

void CUDAImageSolver::solve(std::shared_ptr<SimpleBuffer>   result, const SFSSolverInput& rawSolverInput)
{
    m_solverState.d_x = (float*)result->data();

    SolverInput solverInput;
    solverInput.N = m_imageWidth*m_imageHeight;
    solverInput.width = m_imageWidth;
    solverInput.height = m_imageHeight;

    solverInput.d_targetIntensity   = (float*)rawSolverInput.targetIntensity->data();
    solverInput.d_targetDepth = (float*)rawSolverInput.targetDepth->data();
    solverInput.d_depthMapRefinedLastFrameFloat = (float*)rawSolverInput.previousDepth->data();
    solverInput.d_maskEdgeMap = (unsigned char*)rawSolverInput.maskEdgeMap->data();
    
    cudaMalloc(&solverInput.d_litcoeff, sizeof(float)*9);
    cudaMemcpy(solverInput.d_litcoeff, rawSolverInput.parameters.lightingCoefficients, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    solverInput.deltaTransform = rawSolverInput.parameters.deltaTransform; // transformation to last frame
    solverInput.calibparams.ux = rawSolverInput.parameters.ux;
    solverInput.calibparams.uy = rawSolverInput.parameters.uy;
    solverInput.calibparams.fx = rawSolverInput.parameters.fx;
    solverInput.calibparams.fy = rawSolverInput.parameters.fy;

    SolverParameters parameters;
    parameters.weightFitting = rawSolverInput.parameters.weightFitting;
    parameters.weightShadingStart = rawSolverInput.parameters.weightShadingStart;
    parameters.weightShadingIncrement = rawSolverInput.parameters.weightShadingIncrement;
    parameters.weightShading = rawSolverInput.parameters.weightShadingStart + rawSolverInput.parameters.weightShadingIncrement * rawSolverInput.parameters.nLinIterations; // default to final value
    parameters.weightRegularizer = rawSolverInput.parameters.weightRegularizer;
    parameters.weightBoundary = rawSolverInput.parameters.weightBoundary;
    parameters.weightPrior = rawSolverInput.parameters.weightPrior;
    parameters.nNonLinearIterations = rawSolverInput.parameters.nNonLinearIterations;
    parameters.nLinIterations = rawSolverInput.parameters.nLinIterations;
    parameters.nPatchIterations = rawSolverInput.parameters.nPatchIterations;
	
    ConvergenceAnalysis<float>* ca = NULL;
    solveSFSStub(solverInput, m_solverState, parameters, ca);
}

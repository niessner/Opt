#include "CUDAPatchSolverWarping.h"
#include "PatchSolverWarpingParameters.h"

extern "C" void patchSolveStereoStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters);

CUDAPatchSolverWarping::CUDAPatchSolverWarping(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	const unsigned int N = m_imageWidth*m_imageHeight;
	const unsigned int numberOfVariables = N;

	cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));
}

CUDAPatchSolverWarping::~CUDAPatchSolverWarping()
{
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
}

void CUDAPatchSolverWarping::solveGN(float4* d_image, float4* d_target, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightRegularizer)
{
	m_solverState.d_x = d_image;
	m_solverState.d_mask = d_mask;
	m_solverState.d_target = d_target;

	PatchSolverParameters parameters;
	parameters.weightFitting = weightFitting;
	parameters.weightRegularizer = weightRegularizer;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinearIterations = nLinearIterations;
	parameters.nPatchIterations = nPatchIterations;

	PatchSolverInput solverInput;
	solverInput.N = m_imageWidth*m_imageHeight;
	solverInput.width = m_imageWidth;
	solverInput.height = m_imageHeight;
	
	patchSolveStereoStub(solverInput, m_solverState, parameters);
}

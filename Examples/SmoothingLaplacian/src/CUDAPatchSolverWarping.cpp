#include "CUDAPatchSolverWarping.h"
#include "PatchSolverWarpingParameters.h"

extern "C" void patchSolveStereoStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters);

CUDAPatchSolverWarping::CUDAPatchSolverWarping(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	const unsigned int N = m_imageWidth*m_imageHeight;
	const unsigned int numberOfVariables = N;
}

CUDAPatchSolverWarping::~CUDAPatchSolverWarping()
{
}

void CUDAPatchSolverWarping::solveGN(float* d_image, float* d_target, unsigned int nNonLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightRegularizer)
{
	m_solverState.d_x = d_image;
	m_solverState.d_target = d_target;

	PatchSolverParameters parameters;
	parameters.weightFitting = weightFitting;
	parameters.weightRegularizer = weightRegularizer;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nPatchIterations = nPatchIterations;

	PatchSolverInput solverInput;
	solverInput.N = m_imageWidth*m_imageHeight;
	solverInput.width = m_imageWidth;
	solverInput.height = m_imageHeight;
	
	patchSolveStereoStub(solverInput, m_solverState, parameters);
}

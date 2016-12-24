
#include "CERESWarpingSolver.h"

CERESWarpingSolver::CERESWarpingSolver(unsigned int N) : m_N(N)
{
	const unsigned int numberOfVariables = N;
	
	const unsigned int THREADS_PER_BLOCK = 512; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;
	// TODO: tmpBufferSize on CERES?
	
	// State
	m_solverState.d_delta = new float3[numberOfVariables];
	m_solverState.d_deltaA = new float3[numberOfVariables];
	m_solverState.d_r = new float3[numberOfVariables];
	m_solverState.d_rA = new float3[numberOfVariables];
	m_solverState.d_z = new float3[numberOfVariables];
	m_solverState.d_zA = new float3[numberOfVariables];
	m_solverState.d_p = new float3[numberOfVariables];
	m_solverState.d_pA = new float3[numberOfVariables];
	m_solverState.d_Ap_X = new float3[numberOfVariables];
	m_solverState.d_Ap_A = new float3[numberOfVariables];
	m_solverState.d_scanAlpha = new float[tmpBufferSize];
	m_solverState.d_scanBeta = new float[tmpBufferSize];
	m_solverState.d_rDotzOld = new float[tmpBufferSize];
	m_solverState.d_precondioner = new float3[tmpBufferSize];
	m_solverState.d_precondionerA = new float3[tmpBufferSize];
	m_solverState.d_sumResidual = new float[1];
}

CERESWarpingSolver::~CERESWarpingSolver()
{

}

void CERESWarpingSolver::solve(
	int3 dims,
	float3* d_vertexPosFloat3, 
	float3* d_anglesFloat3, 
	float3* d_vertexPosFloat3Urshape, 
	float3* d_vertexPosTargetFloat3, 
	int nonLinearIter, 
	int linearIter, 
	float weightFit, 
	float weightReg)
{
	/*m_solverState.d_urshape = d_vertexPosFloat3Urshape;
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
	solverInput.dims = dims;

	ImageWarpiungSolveGNStub(solverInput, m_solverState, parameters);*/
}

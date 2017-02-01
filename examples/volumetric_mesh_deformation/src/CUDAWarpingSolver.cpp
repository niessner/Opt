#include "CUDAWarpingSolver.h"
#include "OptSolver.h"

extern "C" double ImageWarpingSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters);	// gauss newton



CUDAWarpingSolver::CUDAWarpingSolver(const std::vector<unsigned int>& dims) : m_dims(dims)
{

    const unsigned int numberOfVariables = dims[0] * dims[1] * dims[2];

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
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)));
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

float sq(float x) { return x*x; }

double CUDAWarpingSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, bool profileSolve, std::vector<SolverIteration>& iters)
{

    m_solverState.d_urshape = getTypedParameterImage<float3>("UrShape", probParams);
    m_solverState.d_a = getTypedParameterImage<float3>("Angle", probParams);
    m_solverState.d_target = getTypedParameterImage<float3>("Constraints", probParams);
    m_solverState.d_x = getTypedParameterImage<float3>("Offset", probParams);


	SolverParameters parameters;
    parameters.weightFitting = sq(getTypedParameter<float>("w_fitSqrt", probParams));
    parameters.weightRegularizer = sq(getTypedParameter<float>("w_regSqrt", probParams));
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nonLinearIterations", solverParams);
    parameters.nLinIterations = getTypedParameter<unsigned int>("linearIterations", solverParams);
	
	SolverInput solverInput;
    solverInput.N = m_dims[0] * m_dims[1] * m_dims[2];
    solverInput.dims = make_int3(m_dims[0] - 1, m_dims[1] - 1, m_dims[2] - 1);

	return ImageWarpingSolveGNStub(solverInput, m_solverState, parameters);
}

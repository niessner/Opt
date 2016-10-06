#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}

#include <cuda_runtime.h>
#include <cudaUtil.h>

extern "C" void reshuffleToFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);
extern "C" void reshuffleFromFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);

class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, unsigned int depth, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t dims[] = { width, height, depth };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		m_width = width;
		m_height = height;
		m_depth = depth;

		CUDA_SAFE_CALL(cudaMalloc(&d_unknown, sizeof(float3)*width*height*depth));

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
		assert(d_unknown);
	}

	~TerraSolverWarping()
	{
		CUDA_SAFE_CALL(cudaFree(d_unknown));


		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}


	void solve(float3* d_x, float3* d_a, float3* d_urshape, float3* d_constraints, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		
		void* problemParams[] = { d_x, d_a, d_urshape, d_constraints, &weightFitSqrt, &weightRegSqrt };
		
		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
	}


	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;

	float3*	d_unknown;
	unsigned int m_width, m_height, m_depth;
};

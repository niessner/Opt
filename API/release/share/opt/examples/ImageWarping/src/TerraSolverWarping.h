#pragma once

#include <cassert>
#include "cudaUtil.h"
extern "C" {
#include "Opt.h"
}

#include <cuda_runtime.h>

class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t dims[] = { width, height };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		m_width = width;
		m_height = height;

		cutilSafeCall(cudaMalloc(&d_unknown, sizeof(float3)*width*height));

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
		assert(d_unknown);
	}

	~TerraSolverWarping()
	{
		cutilSafeCall(cudaFree(d_unknown));


		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}


	void solve(float2* d_x, float* d_a, float2* d_urshape, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations };
		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		
		void* problemParams[] = { d_x, d_a, d_urshape, d_constraints, d_mask, &weightFitSqrt, &weightRegSqrt };
		
		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
	}


	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;

	float3*	d_unknown;
	unsigned int m_width, m_height;
};

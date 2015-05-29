#pragma once

#include <cassert>

extern "C" {
#include "../Opt.h"
}

#include <cuda_runtime.h>
#include <cudaUtil.h>

class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);


		uint64_t strides[] = { 
			width * sizeof(float3), //X (includes uv, a)
			width * sizeof(float2),	//UrShape
			width * sizeof(float2),	//Constraints
			width * sizeof(float)
		};
		uint64_t elemsizes[] = { 
			sizeof(float3), //X (includes uv, a)
			sizeof(float2),	//UrShape
			sizeof(float2),	//Constraints
			sizeof(float)
		};
		uint64_t dims[] = { width, height };

		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, elemsizes, strides, nullptr, nullptr, nullptr);


		CUDA_SAFE_CALL(cudaMalloc(&d_unknown, sizeof(float3)*width*height));

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


	void solve(float2* d_x, float* d_a, float2* d_urshape, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFit, float weightReg)
	{
		solve(d_unknown, d_urshape, d_constraints, d_mask, nNonLinearIterations, nLinearIterations, weightFit, weightReg);
	}

	void solve(float3* d_unknown, float2* d_urshape, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFit, float weightReg)
	{

		void* data[] = {d_unknown, d_urshape, d_constraints, d_mask};
		//last parameter is params
		//Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, list, NULL);

		unsigned int numIter[] = { nNonLinearIterations, nLinearIterations };

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		void* problemParams[] = { &weightFitSqrt, &weightRegSqrt };


		Opt_ProblemInit(m_optimizerState, m_plan, data, NULL, problemParams, (void**)&numIter);
		while (Opt_ProblemStep(m_optimizerState, m_plan, data, NULL, problemParams, NULL));
		//Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, NULL, NULL);
	}

private:

	void reshuffleUnknowsToLocal(float2* d_x, float* d_a) {

	}

	void reshuffleUnknownsFromLocal(float2* d_x, float* d_a) {

	}


	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;

	float3*	d_unknown;
};

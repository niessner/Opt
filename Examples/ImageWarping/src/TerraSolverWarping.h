#pragma once

#include <cassert>

extern "C" {
#include "../Opt.h"
}

#include <cuda_runtime.h>
#include <cudaUtil.h>

extern "C" void reshuffleToFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);
extern "C" void reshuffleFromFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);

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

		m_width = width;
		m_height = height;

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


	void solve(float2* d_x, float* d_a, float2* d_urshape, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{
		reshuffleUnknowsToLocal(d_x, d_a);
		solve(d_unknown, d_urshape, d_constraints, d_mask, nNonLinearIterations, nLinearIterations, nBlockIterations, weightFit, weightReg);
		reshuffleUnknownsFromLocal(d_x, d_a);
	}

	void solve(float3* d_unknown, float2* d_urshape, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);

		void* problemParams[] = { &weightFitSqrt, &weightRegSqrt };
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		void* data[] = { d_unknown, d_urshape, d_constraints, d_mask };
		


		//Opt_ProblemInit(m_optimizerState, m_plan, data, NULL, problemParams, (void**)&numIter);
		//while (Opt_ProblemStep(m_optimizerState, m_plan, data, NULL, problemParams, NULL));
		Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, problemParams, solverParams);
	}

private:

	void reshuffleUnknowsToLocal(float2* d_x, float* d_a) {
		reshuffleToFloat3CUDA(d_x, d_a, d_unknown, m_width, m_height);
	}

	void reshuffleUnknownsFromLocal(float2* d_x, float* d_a) {
		reshuffleFromFloat3CUDA(d_x, d_a, d_unknown, m_width, m_height);
	}


	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;

	float3*	d_unknown;
	unsigned int m_width, m_height;
};

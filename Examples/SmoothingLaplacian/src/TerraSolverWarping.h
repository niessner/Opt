#pragma once

#include <cassert>

extern "C" {
#include "../Opt.h"
}

class TerraSolverWarping {

public:
	TerraSolverWarping(
		unsigned int width,
		unsigned int height,
		const std::string& terraFile = "smoothingLaplacianAD.t", 
		const std::string& optName = "gaussNewtonGPU") 
			: m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);


		uint64_t strides[] = { width * sizeof(float), width * sizeof(float) };
		uint64_t elemsizes[] = { sizeof(float), sizeof(float) };
		uint64_t dims[] = { width, height };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, elemsizes, strides, nullptr, nullptr, nullptr);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~TerraSolverWarping()
	{
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	void solve(float* d_unknown, float* d_target)
	{
		int a = 0;
		void * list[] = { &a };	//TODO fix the bug that we need some 'parameter list'

		void* data[] = {d_unknown, d_target};
		//last parameter is params
		Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, list);
	}

private:


	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;
};

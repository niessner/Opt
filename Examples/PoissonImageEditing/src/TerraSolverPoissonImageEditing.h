#pragma once

#include <cassert>

#include "cutil.h"

extern "C" {
#include "Opt.h"
}

class TerraSolverPoissonImageEditing {

public:
	TerraSolverPoissonImageEditing(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());


		uint32_t dims[] = { width, height };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~TerraSolverPoissonImageEditing()
	{
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	void solve(float4* d_unknown, float4* d_target, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		void* problemParams[] = { d_unknown, d_target, d_mask };

		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
	}

private:
    Opt_State*	m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;
};

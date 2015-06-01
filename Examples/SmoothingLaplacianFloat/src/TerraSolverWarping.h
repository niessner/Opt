#pragma once

#include <cassert>

extern "C" {
#include "../Opt.h"
}

class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);


		uint strides[] = { width * sizeof(float), width * sizeof(float) };
		uint elemsizes[] = { sizeof(float), sizeof(float) };
		uint dims[] = { width, height };
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

	void solve(float* d_unknown, float* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		void* data[] = {d_unknown, d_target};
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		void* problemParams[] = { &weightFit, &weightReg };

		//Opt_ProblemInit(m_optimizerState, m_plan, data, NULL, problemParams, solverParams);
		//while (Opt_ProblemStep(m_optimizerState, m_plan, data, NULL, problemParams, NULL));
		Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, problemParams, solverParams);
	}

private:
	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;
};

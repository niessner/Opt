#pragma once

#include <cassert>

#include "cutil.h"

extern "C" {
#include "Opt.h"
}

class TerraSolver {
public:
	TerraSolver(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t dims[] = { width, height };

		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

    ~TerraSolver()
	{

		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	void solve(float3* d_unknownAlbedo, float* d_unknownIllumination, float3* d_target, float3* d_input, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightRegAlbedo, float weightRegShading, float weightRegChroma, float pNorm)
	{

		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrtAlbedo = sqrt(weightRegAlbedo);
		float weightRegSqrtShading = sqrt(weightRegShading);
		float weightRegSqrtChroma = sqrt(weightRegChroma);

		void* problemParams[] = { &weightFitSqrt, &weightRegSqrtAlbedo, &weightRegSqrtShading, &weightRegSqrtChroma, &pNorm, d_unknownAlbedo, d_target, d_input, d_unknownIllumination };

		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);

        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
	}

    double finalCost() const {
        return m_finalCost;
    }

private:
    double m_finalCost = nan(nullptr);
	Opt_State*	m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;
};

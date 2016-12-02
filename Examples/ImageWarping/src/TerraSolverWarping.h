#pragma once

#include <cassert>

#include "SolverIteration.h"
extern "C" {
#include "Opt.h"
}

#include "Precision.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

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


    void solve(OPT_FLOAT2* d_x, OPT_FLOAT* d_a, OPT_FLOAT2* d_urshape, OPT_FLOAT2* d_constraints, OPT_FLOAT* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg, std::vector<SolverIteration>& iterationSummary)
	{
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		
		void* problemParams[] = { d_x, d_a, d_urshape, d_constraints, d_mask, &weightFitSqrt, &weightRegSqrt };
		
        Timer t;
        t.start();
        Opt_ProblemInit(m_optimizerState, m_plan, problemParams, solverParams);
        cudaDeviceSynchronize();
        t.stop();
        double cost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
        iterationSummary.push_back(SolverIteration(cost, t.getElapsedTimeMS()));
        
        
        t.start();
        while (Opt_ProblemStep(m_optimizerState, m_plan, problemParams, solverParams)) {
            cudaDeviceSynchronize();
            t.stop();
            cost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
            iterationSummary.push_back(SolverIteration(cost, t.getElapsedTimeMS()));
            t.start();
        }
        t.stop();
	}


	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;

	unsigned int m_width, m_height;
};

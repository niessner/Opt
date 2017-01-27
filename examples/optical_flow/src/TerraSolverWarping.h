#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}

#include <cuda_runtime.h>
#include <cudaUtil.h>


class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t strides[] = { 
			width * (uint32_t)sizeof(float2),	//X (offset vectors)
			width * (uint32_t)sizeof(float),	//source
			width * (uint32_t)sizeof(float),	//target
			width * (uint32_t)sizeof(float),	//target DU
			width * (uint32_t)sizeof(float)		//target DV
		};
		uint32_t elemsizes[] = { 
			sizeof(float2),				//X (offset vectors) 
			(uint32_t)sizeof(float),	//source
			(uint32_t)sizeof(float),	//target
			(uint32_t)sizeof(float),	//target DU
			(uint32_t)sizeof(float)		//target DV
		};
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


	void solve(float2* d_unknown, float* d_source, float* d_target, float* d_targetDU, float* d_targetDV, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);

		void* problemParams[] = { &weightFitSqrt, &weightRegSqrt, d_unknown, d_source, d_target, d_targetDU, d_targetDV };
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
 		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
	}

    double finalCost() const {
        return m_finalCost;
    }

private:
    double m_finalCost = nan(nullptr);

	Opt_State*	    m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;

	unsigned int m_width, m_height;
};

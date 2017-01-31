#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}
#include "../../shared/OptUtils.h"
#include "../../shared/CudaArray.h"
#include "../../shared/Precision.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>


//std::vector<MultiDimensionalArray> images, 

class OptSolver;

class NamedParameters {
    friend OptSolver;
    void** data() const {
        return (void**)m_data.data();
    }
protected:
    std::vector<void*> m_data;
    std::vector<std::string> m_names;
};

class OptSolver {

public:
    OptSolver(const std::vector<uint32_t>& dimensions, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());
        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, (unsigned int*)dimensions.data());

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

    ~OptSolver()
	{
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}
	}

    void setProblemParam(std::string name, void* data) {
        m_problemParameters.m_names.push_back(name);
        m_problemParameters.m_data.push_back(data);
    }

    void setSolverParam(std::string name, void* data) {
        m_solverParameters.m_names.push_back(name);
        m_solverParameters.m_data.push_back(data);
    }

    void solve(std::vector<SolverIteration>& iters)
	{
		//void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

        //problemParams = { d_x, d_a, d_urshape, d_constraints, &weightFitSqrt, &weightRegSqrt };
        
        launchProfiledSolve(m_optimizerState, m_plan, m_problemParameters.data(), m_solverParameters.data(), iters);
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
	}

    double finalCost() const {
        return m_finalCost;
    }

    NamedParameters m_solverParameters;
    NamedParameters m_problemParameters;

    double m_finalCost = nan(nullptr);
	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;
};

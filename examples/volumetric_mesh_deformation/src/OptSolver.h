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
#include <algorithm>

class OptSolver;
/** 
    Uses parallel vectors, fairly efficient for small # of parameters.
    If parameter count could be large, consider better approaches 
*/
class NamedParameters {
    friend OptSolver;
public:
    void** data() const {
        return (void**)m_data.data();
    }
    void set(const std::string& name, void* data) {
        auto location = std::find(m_names.begin(), m_names.end(), name);
        if (location == m_names.end()) {
            m_names.push_back(name);
            m_data.push_back(data);
        } else {
            *location = name;
        }
    }
    std::vector<std::string> names() const {
        return m_names;
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

    double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profiledSolve, std::vector<SolverIteration>& iters) {
        if (profiledSolve) {
            launchProfiledSolve(m_optimizerState, m_plan, problemParameters.data(), solverParameters.data(), iters);
        } else {
            Opt_ProblemSolve(m_optimizerState, m_plan, problemParameters.data(), solverParameters.data());
        }
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);
        return m_finalCost;
	}

    double finalCost() const {
        return m_finalCost;
    }

    double m_finalCost = nan(nullptr);
	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;
};

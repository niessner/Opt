#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}
#include "../../shared/OptUtils.h"
#include "../../shared/CudaArray.h"
#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>
#include <algorithm>


/**
    Uses SoA, fairly efficient for small # of parameters.
    If parameter count could be large, consider better approaches
*/
class NamedParameters {
public:
    void** data() const {
        return (void**)m_data.data();
    }
    void set(const std::string& name, void* data) {
        auto location = std::find(m_names.begin(), m_names.end(), name);
        if (location == m_names.end()) {
            m_names.push_back(name);
            m_data.push_back(data);
        }
        else {
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

class SolverBase {
public:
    SolverBase() {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) {
        fprintf(stderr, "No solve implemented\n");
        return m_finalCost;
    }
    double finalCost() const {
        return m_finalCost;
    }
protected:
    double m_finalCost = nan(nullptr);
};

class OptSolver {

public:
    OptSolver(const std::vector<unsigned int>& dimensions, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
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


template<class T> size_t index_of(T element, const std::vector<T>& v) {
    auto location = std::find(v.begin(), v.end(), element);
    if (location != v.end()) {
        return std::distance(v.begin(), location);
    }
    else {
        return -1;
    }
}

template<class T> T* getTypedParameterImage(std::string name, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    return (T*)(solverParameters.data()[i]);
}

// TODO: Error handling
template<class T> void findAndCopyArrayToCPU(std::string name, std::vector<T>& cpuBuffer, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    cutilSafeCall(cudaMemcpy(cpuBuffer.data(), solverParameters.data()[i], sizeof(T)*cpuBuffer.size(), cudaMemcpyDeviceToHost));
}
template<class T> void findAndCopyToArrayFromCPU(std::string name, std::vector<T>& cpuBuffer, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    cutilSafeCall(cudaMemcpy(solverParameters.data()[i], cpuBuffer.data(), sizeof(T)*cpuBuffer.size(), cudaMemcpyHostToDevice));
}
template<class T> T getTypedParameter(std::string name, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    return *(T*)solverParameters.data()[i];
}
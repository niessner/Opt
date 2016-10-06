#pragma once

#include <cassert>

#include "cutil.h"


extern "C" {
#include "Opt.h"
}

class OptImageSolver {

public:
	OptImageSolver(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

        

		uint32_t dims[] = { width, height };

        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~OptImageSolver()
	{
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

    struct IterStruct {
        unsigned int* nIter;
        unsigned int* lIter;
        unsigned int* pIter;
        IterStruct(unsigned int* n, unsigned int* l, unsigned int* p) : nIter(n), lIter(l), pIter(p) {}
    };

    void solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput)
	{
        std::vector<void*> images;
        images.push_back(result->data());
        images.push_back(rawSolverInput.targetDepth->data());
        images.push_back(rawSolverInput.targetIntensity->data());
        images.push_back(rawSolverInput.previousDepth->data());
        images.push_back(rawSolverInput.maskEdgeMap->data()); // row
        images.push_back(((unsigned char*)rawSolverInput.maskEdgeMap->data()) + (result->width() * result->height())); // col

        unsigned int nIter[] = { rawSolverInput.parameters.nNonLinearIterations, rawSolverInput.parameters.nLinIterations, rawSolverInput.parameters.nPatchIterations };
        IterStruct iterStruct(&nIter[0], &nIter[1], &nIter[2]);

        TerraSolverParameterPointers indirectParameters(rawSolverInput.parameters, images);

        Opt_ProblemSolve(m_optimizerState, m_plan, (void**)&indirectParameters, (void**)&iterStruct);
	}

private:
	Opt_State*	m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;
};

#pragma once
#include "Precision.h"
#include <cassert>
#include <stdint.h>
extern "C" {
#include "Opt.h"
}
#include "CudaArray.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>


class TerraSolver {

public:
    TerraSolver(uint32_t dataPointCount, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());
        uint32_t dims[] = { dataPointCount, 1 };
        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);		

        // Create graph
        m_edgeCount = (int)dataPointCount;
        std::vector<int> yCoords;

        for (int y = 0; y < m_edgeCount; ++y) {
            yCoords.push_back(0);
        }

        d_headY.update(yCoords.data(), yCoords.size());
        d_tailY.update(yCoords.data(), yCoords.size());

        // Convert to our edge format
        std::vector<int> h_headX;
        std::vector<int> h_tailX;
        for (int headX = 0; headX < m_edgeCount; ++headX) {
            h_headX.push_back(headX);
            h_tailX.push_back(0);
        }
        d_headX.update(h_headX.data(), h_headX.size());
        d_tailX.update(h_tailX.data(), h_tailX.size());

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


    void solve(OPT_UNKNOWNS* d_x, OPT_FLOAT2* d_data, unsigned int nNonLinearIterations, unsigned int nLinearIterations)
	{
        unsigned int nBlockIterations = 0;
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
        void* problemParams[] = { d_x, d_data, &m_edgeCount, d_headX.data(), d_headY.data(), d_tailX.data(), d_tailY.data() };
		
		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
	}

	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;
    
    CudaArray<int> d_headX;
    CudaArray<int> d_headY;
    CudaArray<int> d_tailX;
    CudaArray<int> d_tailY;
	int m_edgeCount;
};

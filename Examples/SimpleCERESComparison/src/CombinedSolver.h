#pragma once

#include "Precision.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>
#include <iostream>

#include "config.h"

#include "TerraSolver.h"
#include "CeresSolver.h"
#include <vector>
static bool useOptGN = true;
static bool useCeres = true;
static bool useOptLM = true;

class CombinedSolver {
public:
	CombinedSolver(const NLLSProblem &problem, std::vector<double2> dataPoints) {

		std::string optProblemFilename = "terra/" + problem.baseName + ".t";
		if (useProblemDefault) optProblemFilename = "curveFitting.t";
		
        m_ceresDataPoints = dataPoints;
        m_functionParametersGuess = problem.startingPoint;
        std::vector<OPT_FLOAT2> dataPointsFloat(dataPoints.size());
        for (int i = 0; i < dataPoints.size(); ++i) {
            dataPointsFloat[i].x = (OPT_FLOAT)dataPoints[i].x;
            dataPointsFloat[i].y = (OPT_FLOAT)dataPoints[i].y;
        }
        d_dataPoints.update(dataPointsFloat);
		resetGPU();
        if (useOptLM) {
            m_solverOptLM = new TerraSolver((uint32_t)dataPoints.size(), optProblemFilename, "LMGPU");
        }

        if (useOptGN) {
            m_solverOptGN = new TerraSolver((uint32_t)dataPoints.size(), optProblemFilename, "gaussNewtonGPU");
        }

        if (useCeres) {
            m_solverCeres = new CeresSolver(dataPoints.size());
        }

	}

	void resetGPU() {
        std::vector<OPT_UNKNOWNS> unknowns(1);
		for (int i = 0; i < maxUnknownCount; i++)
		{
			*((OPT_FLOAT*)&unknowns[0] + i) = (OPT_FLOAT)((double*)&m_functionParametersGuess + i)[0];
		}
        d_functionParameters.update(unknowns);
	}

    void copyResultToCPU() {
		std::vector<OPT_UNKNOWNS> unknowns;
        d_functionParameters.readBack(unknowns);
		for (int i = 0; i < maxUnknownCount; i++)
		{
			*((double*)&m_functionParameters + i) = ((OPT_FLOAT*)&unknowns[0] + i)[0];
		}
        //m_functionParameters.x = unknowns[0].x;
        //m_functionParameters.y = unknowns[0].y;
    }

	UNKNOWNS solve(const NLLSProblem &problem) {
		std::cout << "*** Solving " << problem.baseName << std::endl;
        uint nonLinearIter = 10000;
        uint linearIter = problem.unknownCount + 3;
        if (useOptLM) {
			resetGPU();
            m_optLMIters = m_solverOptLM->solve(d_functionParameters.data(), d_dataPoints.data(), nonLinearIter, linearIter);
			copyResultToCPU();
            m_optLMResult = m_functionParameters;
		}
		if (useCeres) {
            m_functionParameters = m_functionParametersGuess;
			m_ceresIters = m_solverCeres->solve(problem, &m_functionParameters, m_ceresDataPoints.data());
			m_ceresResult = m_functionParameters;
		}

        nonLinearIter = (uint)min(m_ceresIters.size(), m_optLMIters.size());

        if (useOptGN) {
            resetGPU();
            m_optGNIters = m_solverOptGN->solve(d_functionParameters.data(), d_dataPoints.data(), nonLinearIter, linearIter);
            copyResultToCPU();
            m_optGNResult = m_functionParameters;
        }

        return m_functionParameters;
	}

    ~CombinedSolver() {
        if (useOptLM) {
            delete m_solverOptLM;
        }
        if (useOptGN) {
            delete m_solverOptGN;
        }

        if (useCeres) {
            delete m_solverCeres;
        }
    }

	UNKNOWNS m_optLMResult;
    UNKNOWNS m_optGNResult;
	UNKNOWNS m_ceresResult;

	std::vector<SolverIteration> m_ceresIters;
	std::vector<SolverIteration> m_optLMIters;
    std::vector<SolverIteration> m_optGNIters;

private:
    std::vector<double2> m_ceresDataPoints;
	UNKNOWNS m_functionParameters;

	UNKNOWNS m_functionParametersGuess;
    CudaArray<OPT_FLOAT2> d_dataPoints;
	CudaArray<OPT_UNKNOWNS> d_functionParameters;
	

	TerraSolver*		m_solverOptLM;
    TerraSolver*		m_solverOptGN;
	CeresSolver*		m_solverCeres;
};

#pragma once

#include "Precision.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "config.h"

#include "TerraSolver.h"
#include "CeresSolver.h"
#include <vector>
static bool useOpt = true;
static bool useCeres = true;
static bool useOptLM = true;

class CombinedSolver {
public:
	CombinedSolver(const NLLSProblem &problem, std::vector<double2> dataPoints) {

		std::string optProblemFilename = problem.baseName + ".t";
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

        if (useOpt) {
			m_solverOpt = new TerraSolver((uint32_t)dataPoints.size(), optProblemFilename, useOptLM ? "LMGPU" : "gaussNewtonGPU");
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
        //unknowns[0].x = (OPT_FLOAT)m_functionParametersGuess.x;
        //unknowns[0].y = (OPT_FLOAT)m_functionParametersGuess.y;
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
        uint nonLinearIter = 1000;
        uint linearIter = 1000;
		if (useOpt) {
			resetGPU();
            m_solverOpt->solve(d_functionParameters.data(), d_dataPoints.data(), nonLinearIter, linearIter);
			copyResultToCPU();
			m_optResult = m_functionParameters;
		}

		if (useCeres) {
            m_functionParameters = m_functionParametersGuess;
			m_solverCeres->solve(problem, &m_functionParameters, m_ceresDataPoints.data());
			m_ceresResult = m_functionParameters;
		}

        return m_functionParameters;
	}

	UNKNOWNS m_optResult;
	UNKNOWNS m_ceresResult;

private:
    std::vector<double2> m_ceresDataPoints;
	UNKNOWNS m_functionParameters;

	UNKNOWNS m_functionParametersGuess;
    CudaArray<OPT_FLOAT2> d_dataPoints;
	CudaArray<OPT_UNKNOWNS> d_functionParameters;
	

	TerraSolver*		m_solverOpt;
	CeresSolver*		m_solverCeres;
};

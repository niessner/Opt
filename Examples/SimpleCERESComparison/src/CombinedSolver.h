#pragma once

#include "Precision.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraSolver.h"
#include "CeresSolver.h"
#include <vector>
static bool useOpt = true;
static bool useCeres = true;

class CombinedSolver {
public:
    CombinedSolver(double2 functionParameterGuess, std::vector<double2> dataPoints) {
        m_ceresDataPoints = dataPoints;
        m_functionParametersGuess = functionParameterGuess;
        std::vector<OPT_FLOAT2> dataPointsFloat(dataPoints.size());
        for (int i = 0; i < dataPoints.size(); ++i) {
            dataPointsFloat[i].x = (OPT_FLOAT)dataPoints[i].x;
            dataPointsFloat[i].y = (OPT_FLOAT)dataPoints[i].y;
        }
        d_dataPoints.update(dataPointsFloat);
		resetGPU();


        if (useOpt) {
            m_solverOpt = new TerraSolver((uint32_t)dataPoints.size(), "curveFitting.t", "gaussNewtonGPU");
		}

        if (useCeres) {
            m_solverCeres = new CeresSolver(dataPoints.size());
        }

	}

	void resetGPU() {
        std::vector<OPT_FLOAT2> unknowns(1);
        unknowns[0].x = (OPT_FLOAT)m_functionParametersGuess.x;
        unknowns[0].y = (OPT_FLOAT)m_functionParametersGuess.y;
        d_functionParameters.update(unknowns);
	}

    void copyResultToCPU() {
        std::vector<OPT_FLOAT2> unknowns;
        d_functionParameters.readBack(unknowns);
        m_functionParameters.x = unknowns[0].x;
        m_functionParameters.y = unknowns[0].y;
    }

	double2 solve() {
        uint nonLinearIter = 10;
        uint linearIter = 1000;
		if (useOpt) {
			resetGPU();
            m_solverOpt->solve(d_functionParameters.data(), d_dataPoints.data(), nonLinearIter, linearIter);
			copyResultToCPU();
		}

		if (useCeres) {
            m_solverCeres->solve(&m_functionParameters, m_ceresDataPoints.data());
		}

        return m_functionParameters;
	}

	

private:
    std::vector<double2> m_ceresDataPoints;
    double2 m_functionParameters;

    double2 m_functionParametersGuess;
    CudaArray<OPT_FLOAT2> d_dataPoints;
    CudaArray<OPT_FLOAT2> d_functionParameters;
	

	TerraSolver*		m_solverOpt;
	CeresSolver*		m_solverCeres;
};

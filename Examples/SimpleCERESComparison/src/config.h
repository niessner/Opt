
#pragma once

#include <string>
#include <assert.h>
#include <vector>
#include "Precision.h"
struct NLLSProblem
{
    NLLSProblem(std::string _baseName, int _unknownCount, const std::vector<double>& _startingPoint, const std::vector<double>& _trueSolution)
	{
		baseName = _baseName;
		unknownCount = _unknownCount;
        memset(&startingPoint.vals, 0, sizeof(double9));
        memset(&trueSolution.vals, 0, sizeof(double9));
        assert(_startingPoint.size() == _trueSolution.size());
        for (int i = 0; i < _startingPoint.size(); ++i) {
            startingPoint.vals[i] = _startingPoint[i];
            trueSolution.vals[i] = _trueSolution[i];
        }
	}
	std::string baseName;
	int unknownCount;
	double9 startingPoint, trueSolution;
};

struct SolverIteration
{
	SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
	double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};

const bool useProblemDefault = false; //2 unknowns -- Michael's original curveFitting.t code

// Corresponds to the ENSO problem
const int maxUnknownCount = 9;

#define OPT_UNKNOWNS OPT_FLOAT9

#define UNKNOWNS double9
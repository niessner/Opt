
#pragma once

#include <string>

struct NLLSProblem
{
	NLLSProblem(std::string _baseName, int _unknownCount, double4 _startingPoint, double4 _trueSolution)
	{
		baseName = _baseName;
		unknownCount = _unknownCount;
		trueSolution = _trueSolution;
		startingPoint = _startingPoint;
	}
	std::string baseName;
	int unknownCount;
	double4 startingPoint, trueSolution;
};

struct SolverIteration
{
	SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
	double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};

const bool useProblemDefault = false; //2 unknowns -- Michael's original curveFitting.t code

const int maxUnknownCount = 4;

//#define OPT_UNKNOWNS OPT_FLOAT2
#define OPT_UNKNOWNS OPT_FLOAT4

//#define UNKNOWNS double2
#define UNKNOWNS double4
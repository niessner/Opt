
#pragma once

#include <string>

struct NLLSProblem
{
	NLLSProblem(std::string _baseName, int _unknownCount, double3 _startingPoint, double3 _trueSolution)
	{
		baseName = _baseName;
		unknownCount = _unknownCount;
		trueSolution = _trueSolution;
		startingPoint = _startingPoint;
	}
	std::string baseName;
	int unknownCount;
	double3 startingPoint, trueSolution;
};

const bool useProblemDefault = false; //2 unknowns -- Michael's original curveFitting.t code

const int maxUnknownCount = 3;

//#define OPT_UNKNOWNS OPT_FLOAT2
#define OPT_UNKNOWNS OPT_FLOAT3

//#define UNKNOWNS double2
#define UNKNOWNS double3
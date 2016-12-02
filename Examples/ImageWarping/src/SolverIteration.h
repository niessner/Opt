#pragma once
#include <limits>



struct SolverIteration
{
    SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
    double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};
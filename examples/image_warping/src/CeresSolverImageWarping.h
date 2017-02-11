#pragma once
#include "../../shared/Precision.h"
#include "../../shared/CeresSolverBase.h"
class CeresSolverWarping : public CeresSolverBase {
public:
    CeresSolverWarping(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;
};
#ifndef USE_CERES
CeresSolverWarping::CeresSolverWarping(const std::vector<unsigned int>&) {}
inline double CeresSolverWarping::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter)
{
    return nan(nullptr);
}
#endif
#pragma once

#include <cuda_runtime.h>

#include "mLibInclude.h"
#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"

#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"

#include "OptSolver.h"


#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;

#ifdef _WIN32
#ifndef USE_CERES
#define USE_CERES
#endif
#endif

class CeresSolverBase : public SolverBase {
public:
    CeresSolverBase(const std::vector<unsigned int>& dims) : m_dims(dims) {}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override {
        fprintf(stderr, "No Ceres solve implemented\n");
        return m_finalCost;
    }

protected:
    double launchProfiledSolveAndSummary(const std::unique_ptr<Solver::Options>& options, Problem* problem, bool profileSolve, std::vector<SolverIteration>& iter);
    std::unique_ptr<Solver::Options> initializeOptions(const NamedParameters& solverParameters) const;
    std::vector<unsigned int> m_dims;
};


class CeresSolver : public CeresSolverBase
{
	public:
        CeresSolver(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;

};
#ifndef USE_CERES
CeresSolver::CeresSolver(unsigned int, unsigned int, unsigned int) {}
CeresSolver::~CeresSolver() {}
void CeresSolver::solve(float3*, float3*, float3*, float3*, float, float, std::vector<SolverIteration>&) {}
#endif
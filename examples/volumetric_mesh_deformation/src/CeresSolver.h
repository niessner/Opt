#pragma once

#include <cuda_runtime.h>

#include "mLibInclude.h"
#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"

#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"

#include "../../shared/CeresSolverBase.h"


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


class CeresSolver : public CeresSolverBase
{
	public:
        CeresSolver(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;
};
#ifndef USE_CERES
CeresSolver::CeresSolver(const std::vector<unsigned int>&) {}
virtual double CeresSolver::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) {}
#endif
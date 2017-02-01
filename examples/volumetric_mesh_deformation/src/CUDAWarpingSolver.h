#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "OptSolver.h"
#include <vector>

class CUDAWarpingSolver : public SolverBase
{
	public:
		CUDAWarpingSolver(const std::vector<unsigned int>& dims);
		~CUDAWarpingSolver();

        double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters);
		
	private:

		SolverState	m_solverState;
        std::vector<unsigned int> m_dims;
};

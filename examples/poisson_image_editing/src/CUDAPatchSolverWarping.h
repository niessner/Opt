#pragma once

#include <cuda_runtime.h>
//#include <cuda_d3d11_interop.h>

#include "../../cudaUtil.h"
#include "PatchSolverWarpingState.h"
#include "../../shared/SolverBase.h"

class CUDAPatchSolverWarping : public SolverBase
{
	public:
        CUDAPatchSolverWarping(const std::vector<unsigned int>& dims);
        ~CUDAPatchSolverWarping();

        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;

	private:

		PatchSolverState m_solverState;
        std::vector<unsigned int> m_dims;
};

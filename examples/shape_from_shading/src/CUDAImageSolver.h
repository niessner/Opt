#pragma once

#include <cuda_runtime.h>

#include "../../shared/cudaUtil.h"
#include "SFSSolverParameters.h"
#include "SFSSolverState.h"

#include <memory>
#include "SimpleBuffer.h"
#include "SFSSolverInput.h"
class CUDAImageSolver
{
	public:
		CUDAImageSolver(unsigned int imageWidth, unsigned int imageHeight);
		~CUDAImageSolver();

        void solve(std::shared_ptr<SimpleBuffer>   result, const SFSSolverInput& rawSolverInput);
		
	private:

		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

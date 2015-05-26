#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"



class CUDAWarpingSolver
{
	public:
		CUDAWarpingSolver(unsigned int imageWidth, unsigned int imageHeight);
		~CUDAWarpingSolver();

		void solveGN(float* d_image, float* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer);
		
	private:

		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

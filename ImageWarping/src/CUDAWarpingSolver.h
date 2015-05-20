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

		//! gauss newton
		void solveGN(float2* d_urshape, float2* d_warpField, float* d_warpAngles, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer);
		
	private:

		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

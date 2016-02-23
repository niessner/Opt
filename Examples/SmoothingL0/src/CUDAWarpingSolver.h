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

		void solveGN(float3* d_image, float3* d_target, float3* d_auxFloat3CM, float3* d_auxFloat3CP, float3* d_auxFloat3MC, float3* d_auxFloat3PC, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer, float weightBeta);
		
	private:

		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

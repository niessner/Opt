#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "LaplacianSolverParameters.h"
#include "LaplacianSolverState.h"


class CUDALaplacianSolver
{
	public:
		CUDALaplacianSolver(unsigned int imageWidth, unsigned int imageHeight);
		~CUDALaplacianSolver();

		//! gauss newton
		void solveGN(float* d_targetDepth, float* d_result, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer);
		
		//! gradient decent
		void solveGD(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer);
	private:

		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};


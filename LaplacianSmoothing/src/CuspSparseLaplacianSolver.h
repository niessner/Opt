#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "LaplacianSolverState.h"

class CuspSparseLaplacianSolver
{
	public:
		CuspSparseLaplacianSolver(unsigned int imageWidth, unsigned int imageHeight);
		~CuspSparseLaplacianSolver();

		//! gauss newton
		void solvePCG(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer);
		
	private:

		void genLaplace(int *row_ptr, int *col_ind, float *val, int N, int nz, float *rhs, float wFit, float wReg, float* target);

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

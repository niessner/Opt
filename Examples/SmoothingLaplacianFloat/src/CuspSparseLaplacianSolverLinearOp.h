#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"

class CuspSparseLaplacianSolverLinearOp
{
	public:
		CuspSparseLaplacianSolverLinearOp(unsigned int imageWidth, unsigned int imageHeight);
		~CuspSparseLaplacianSolverLinearOp();

		//! gauss newton
		void solvePCG(float* d_image, float* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer);
		
	private:

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

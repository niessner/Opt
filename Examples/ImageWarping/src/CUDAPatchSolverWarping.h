#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "PatchSolverWarpingState.h"

class CUDAPatchSolverWarping
{
	public:

		CUDAPatchSolverWarping(unsigned int imageWidth, unsigned int imageHeight);
		~CUDAPatchSolverWarping();

		void solveGN(float2* d_urshape, float2* d_warpField, float* d_warpAngles, float2* d_constraints, float* d_mask, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightRegularizer);
			
	private:

		PatchSolverState m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

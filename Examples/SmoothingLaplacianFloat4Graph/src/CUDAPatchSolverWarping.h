#pragma once

#include <cuda_runtime.h>
//#include <cuda_d3d11_interop.h>

#include "../../cudaUtil.h"
#include "PatchSolverWarpingState.h"

class CUDAPatchSolverWarping
{
	public:

		CUDAPatchSolverWarping(unsigned int imageWidth, unsigned int imageHeight);
		~CUDAPatchSolverWarping();

		void solveGN(float4* d_image, float4* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightRegularizer);
			
	private:

		PatchSolverState m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

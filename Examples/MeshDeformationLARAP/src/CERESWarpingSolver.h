#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"

#include "../../shared/Precision.h"

class CERESWarpingSolver
{
	public:
		CERESWarpingSolver(unsigned int width, unsigned int height, unsigned int depth);
		~CERESWarpingSolver();

		void solve(float3* d_x, float3* d_a, float3* d_urshape, float3* d_constraints, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg);
		
	private:

		vec3d*	d_unknown;
		unsigned int m_width, m_height, m_depth;
};

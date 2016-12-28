#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"

#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"

class CERESWarpingSolver
{
	public:
		CERESWarpingSolver(unsigned int width, unsigned int height, unsigned int depth);
		~CERESWarpingSolver();

		void solve(float3* d_vertexPosFloat3,
				   float3* d_anglesFloat3,
				   float3* d_vertexPosFloat3Urshape,
				   float3* d_vertexPosTargetFloat3,
				   float weightFit,
                   float weightReg,
                   std::vector<SolverIteration>& iter);

        double finalCost() const {
            return m_finalCost;
        }


	private:
		vec3d* vertexPosDouble3;
		vec3d* anglesDouble3;

		float3* h_vertexPosFloat3;
		float3* h_anglesFloat3;
		float3* h_vertexPosFloat3Urshape;
		float3* h_vertexPosTargetFloat3;

		unsigned int m_width, m_height, m_depth;
		unsigned int voxelCount;
        double m_finalCost = nan(nullptr);
};

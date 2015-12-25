#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"


class CUDAWarpingSolver
{
	public:
		CUDAWarpingSolver(unsigned int N);
		~CUDAWarpingSolver();

		void solveGN(float3* d_vertexPosFloat3, float3* d_anglesFloat3, float3* d_vertexPosFloat3Urshape, int* d_numNeighbours, int* d_neighbourIdx, int* d_neighbourOffset, float3* d_vertexPosTargetFloat3, int nonLinearIter, int linearIter, float weightFit, float weightReg);
		
	private:

		SolverState	m_solverState;

		unsigned int m_N;
};

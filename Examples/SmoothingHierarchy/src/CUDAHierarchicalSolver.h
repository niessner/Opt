#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "../../cudaUtil.h"
#include "CUDAWarpingSolver.h"

extern "C" void resampleFloat3Map(float3* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float3* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);

class CUDAHierarchicalSolver
{
	public:

		CUDAHierarchicalSolver(std::vector<unsigned int>& optimizationWidthAtLevel, std::vector<unsigned int>& optimizationHeightAtLevel, unsigned int numLevels)
		{
			allocateHierarchy(optimizationWidthAtLevel, optimizationHeightAtLevel, numLevels);
		}

		~CUDAHierarchicalSolver()
		{
			deallocateHierarchy();
		}

		void allocateHierarchy(std::vector<unsigned int>& optimizationWidthAtLevel, std::vector<unsigned int>& optimizationHeightAtLevel, unsigned int numLevels)
		{
			unsigned int originalWidth  = optimizationWidthAtLevel [0];
			unsigned int originalHeight = optimizationHeightAtLevel[0];

			m_widthAtLevel .resize(numLevels);
			m_heightAtLevel.resize(numLevels);

			m_solvers.resize(numLevels);
			
			m_targetAtLevel.resize(numLevels);
			m_imageAtLevel.resize(numLevels);
			
			for(unsigned int i = 0; i < numLevels; i++)
			{
				m_widthAtLevel[i]  = optimizationWidthAtLevel[i];
				m_heightAtLevel[i] = optimizationHeightAtLevel[i];

				m_solvers[i] = new CUDAWarpingSolver(m_widthAtLevel[i], m_heightAtLevel[i]);

				if(i != 0)
				{
					cutilSafeCall(cudaMalloc(&(m_targetAtLevel[i]),		 m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float3)));
					cutilSafeCall(cudaMalloc(&(m_imageAtLevel[i]),		 m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float3)));
				}
			}
		}

		void deallocateHierarchy()
		{
			for(unsigned int i = 0; i < m_solvers.size(); i++)
			{
				delete m_solvers[i];
			}

			for(unsigned int i = 0; i < m_targetAtLevel.size(); i++)
			{
				if (i != 0) cutilSafeCall(cudaFree(m_targetAtLevel[i]));
				if (i != 0) cutilSafeCall(cudaFree(m_imageAtLevel[i]));
			}
		}
		
		void solve(float3* d_image, float3* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer)
		{			
			m_targetAtLevel[0] = d_target;
			m_imageAtLevel[0] = d_image;
	
			// Compute restriction of input data
			unsigned int numLevels = (unsigned int)m_targetAtLevel.size();
			for (unsigned int i = 0; i < numLevels-1; i++)
			{
				resampleFloat3Map(m_targetAtLevel[i + 1], m_widthAtLevel[i + 1], m_heightAtLevel[i + 1], m_targetAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i]);
				resampleFloat3Map(m_imageAtLevel[i + 1] , m_widthAtLevel[i + 1], m_heightAtLevel[i + 1], m_imageAtLevel[i] , m_widthAtLevel[i], m_heightAtLevel[i]);
			}

			// Solve coarse to fine
			for (int i = ((int)numLevels) - 1; i >= 0; i--)
			{
				m_solvers[i]->solveGN(m_imageAtLevel[i], m_targetAtLevel[i], nNonLinearIterations, nLinearIterations, weightFitting, weightRegularizer);
		
				if (i != 0)
				{
					resampleFloat3Map(m_imageAtLevel[i - 1], m_widthAtLevel[i - 1], m_heightAtLevel[i - 1], m_imageAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i]);
				}
			}
		}

	private:

		std::vector<CUDAWarpingSolver*> m_solvers;

		std::vector<unsigned int> m_widthAtLevel;
		std::vector<unsigned int> m_heightAtLevel;

		std::vector<float3*> m_targetAtLevel;
		std::vector<float3*> m_imageAtLevel;

		std::vector<float3*> m_auxFloat3CMAtLevel;
		std::vector<float3*> m_auxFloat3CPAtLevel;
		std::vector<float3*> m_auxFloat3MCAtLevel;
		std::vector<float3*> m_auxFloat3PCAtLevel;
};

#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>
#include "../../shared/Precision.h"
#include "CUDAWarpingSolver.h"
#include "TerraSolver.h"

#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"


class ImageWarping
{
	public:
	
		ImageWarping(const ColorImageR32G32B32A32& image)
		{
			m_image = image;

            cutilSafeCall(cudaMalloc(&d_imageFloat3, sizeof(OPT_FLOAT3)*m_image.getWidth()*m_image.getHeight()));
            cutilSafeCall(cudaMalloc(&d_targetFloat3, sizeof(OPT_FLOAT3)*m_image.getWidth()*m_image.getHeight()));

			resetGPUMemory();

			m_warpingSolver = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
            m_terraSolver = new TerraSolver(m_image.getWidth(), m_image.getHeight(), "SmoothingLaplacianFloat3AD.t", "gaussNewtonGPU");
		}

		void resetGPUMemory()
		{
            OPT_FLOAT3* h_imageFloat3 = new OPT_FLOAT3[m_image.getWidth()*m_image.getHeight()];

			for (unsigned int i = 0; i < m_image.getHeight(); i++)
			{
				for (unsigned int j = 0; j < m_image.getWidth(); j++)
				{
					ml::vec4f v = m_image(j, i);
#if DOUBLE_PRECISION_OPT
                    h_imageFloat3[i*m_image.getWidth() + j] = make_double3(v.x, v.y, v.z);
#else
					h_imageFloat3[i*m_image.getWidth() + j] = make_float3(v.x, v.y, v.z);
#endif
				}
			}

            cutilSafeCall(cudaMemcpy(d_imageFloat3, h_imageFloat3, sizeof(OPT_FLOAT3)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_targetFloat3, h_imageFloat3, sizeof(OPT_FLOAT3)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));

			delete h_imageFloat3;
		}

		~ImageWarping()
		{
			cutilSafeCall(cudaFree(d_imageFloat3));
			cutilSafeCall(cudaFree(d_targetFloat3));

			SAFE_DELETE(m_warpingSolver);
		}

		ColorImageR32G32B32A32* solve()
		{
			float weightFit  = 1.0f;
			float weightReg  = 800.0f;
			float pNorm = 1.0f;

			m_params.nonLinearIter = 50;
			m_params.linearIter = 40;
            m_params.useCUDA = !OPT_DOUBLE_PRECISION;

#if !OPT_DOUBLE_PRECISION
			std::cout << "\n\nCUDA" << std::endl;
			resetGPUMemory();
			m_warpingSolver->solveGN(d_imageFloat3, d_targetFloat3, m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg, pNorm);
			copyResultToCPUFromFloat3();
#endif
            
            std::cout << "\n\nOPT" << std::endl;
            resetGPUMemory();
            m_terraSolver->solve(d_imageFloat3, d_targetFloat3, m_params.nonLinearIter, m_params.linearIter, 0, weightFit, weightReg, pNorm);
            copyResultToCPUFromFloat3();
			
            reportFinalCosts("Smoothing Laplacian Lp", m_params, m_terraSolver->finalCost(), nan(nullptr), nan(nullptr));

			return &m_result;
		}

		void copyResultToCPUFromFloat3()
		{
			m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());

            OPT_FLOAT3* h_imageFloat3 = new OPT_FLOAT3[m_image.getWidth()*m_image.getHeight()];
            cutilSafeCall(cudaMemcpy(h_imageFloat3, d_imageFloat3, sizeof(OPT_FLOAT3)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

			for (unsigned int i = 0; i < m_image.getHeight(); i++)
			{
				for (unsigned int j = 0; j < m_image.getWidth(); j++)
				{
                    OPT_FLOAT3 v = h_imageFloat3[i*m_image.getWidth() + j];
					m_result(j, i) = vec4f(v.x, v.y, v.z, 1.0f);
				}
			}

			delete h_imageFloat3;
		}

	private:

		ColorImageR32G32B32A32 m_result;
		ColorImageR32G32B32A32 m_image;
	
		OPT_FLOAT3*	d_imageFloat3;
        OPT_FLOAT3* d_targetFloat3;

        CombinedSolverParameters m_params;
		CUDAWarpingSolver*		m_warpingSolver;
        TerraSolver* m_terraSolver;
        
};

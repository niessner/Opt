#pragma once

#define RUN_CUDA 1
#define RUN_TERRA 0
#define RUN_OPT 1

#define RUN_CUDA_BLOCK 0
#define RUN_TERRA_BLOCK 0
#define RUN_OPT_BLOCK 0

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverPoissonImageEditing.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32G32B32A32& image, const ColorImageR32G32B32A32& image1, const ColorImageR32& imageMask)
	{
		m_image = image;
		m_image1 = image1;
		m_imageMask = imageMask;

		cutilSafeCall(cudaMalloc(&d_image,	sizeof(float4)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_target, sizeof(float4)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_mask,	sizeof(float) *m_image.getWidth()*m_image.getHeight()));

		resetGPUMemory();

		m_warpingSolver = NULL;
		m_terraSolver = NULL;
		m_optSolver = NULL;

		m_warpingSolverPatch = NULL;
		m_terraBlockSolver = NULL;
		m_optBlockSolver = NULL;

		//non-blocked solvers
#if RUN_CUDA
		m_warpingSolver		 = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());	
#endif
#if RUN_TERRA
		m_terraSolver = new TerraSolverPoissonImageEditing(m_image.getWidth(), m_image.getHeight(), "PoissonImageEditing.t", "gaussNewtonGPU");
#endif
#ifdef RUN_OPT
		m_optSolver = new TerraSolverPoissonImageEditing(m_image.getWidth(), m_image.getHeight(), "PoissonImageEditingAD.t", "gaussNewtonGPU");
#endif

		//blocked solvers
#if RUN_CUDA_BLOCK
		m_warpingSolverPatch = new CUDAPatchSolverWarping(m_image.getWidth(), m_image.getHeight());
#endif
#if RUN_TERRA_BLOCK
		m_terraBlockSolver = new TerraSolverPoissonImageEditing(m_image.getWidth(), m_image.getHeight(), "PoissonImageEditing.t", "gaussNewtonBlockGPU");
#endif
#if RUN_OPT_BLOCK
		m_optBlockSolver = new TerraSolverPoissonImageEditing(m_image.getWidth(), m_image.getHeight(), "PoissonImageEditingAD.t", "gaussNewtonBlockGPU");
#endif
		
	}

	void resetGPUMemory()
	{
		float4* h_image = new float4[m_image.getWidth()*m_image.getHeight()];
		float4* h_target = new float4[m_image.getWidth()*m_image.getHeight()];
		float*  h_mask = new float[m_image.getWidth()*m_image.getHeight()];

		for (unsigned int i = 0; i < m_image1.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image1.getWidth(); j++)
			{
				ml::vec4f v = m_image(j, i);
				h_image[i*m_image.getWidth() + j] = make_float4(v.x, v.y, v.z, 255);

				ml::vec4f t = m_image1(j, i);
				h_target[i*m_image.getWidth() + j] = make_float4(t.x, t.y, t.z, 255);

				if (m_imageMask(j, i) == 255) h_mask[i*m_image.getWidth() + j] = 0;
				else						  h_mask[i*m_image.getWidth() + j] = 255;
			}
		}

		cutilSafeCall(cudaMemcpy(d_image, h_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_target, h_target, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_mask, h_mask, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_mask;
		delete h_image;
		delete h_target;
	}

	~ImageWarping()
	{
		cutilSafeCall(cudaFree(d_image));
		cutilSafeCall(cudaFree(d_target));
		cutilSafeCall(cudaFree(d_mask));

		SAFE_DELETE(m_warpingSolver);
		SAFE_DELETE(m_terraSolver);
		SAFE_DELETE(m_optSolver);

		SAFE_DELETE(m_warpingSolverPatch);
		SAFE_DELETE(m_terraBlockSolver);
		SAFE_DELETE(m_optBlockSolver);
	}

	ColorImageR32G32B32A32* solve()
	{
		//TODO
		float weightFit = 0.0f; // not used
		float weightReg = 0.0f; // not used
		
		unsigned int nonLinearIter = 5;
		unsigned int linearIter = 250;
		unsigned int patchIter = 16;
		
#if RUN_CUDA
		std::cout << "=======CUDA=======" << std::endl;
		resetGPUMemory();
		m_warpingSolver->solveGN(d_image, d_target, d_mask, nonLinearIter, linearIter, weightFit, weightReg);		
		copyResultToCPU();
#endif
#if RUN_TERRA
		std::cout << "\n\n========TERRA========" << std::endl;
		resetGPUMemory();
		m_terraSolver->solve(d_image, d_target, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		copyResultToCPU();
#endif
#if RUN_OPT
		std::cout << "\n\n========OPT========" << std::endl;
		resetGPUMemory();
		m_optSolver->solve(d_image, d_target, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		copyResultToCPU();
#endif



#if RUN_CUDA_BLOCK
		std::cout << "======CUDA_BLOCK====" << std::endl;
		resetGPUMemory();
		m_warpingSolverPatch->solveGN(d_image, d_target, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		copyResultToCPU();
#endif
#if RUN_TERRA_BLOCK
		std::cout << "\n\n======TERRA_BLOCK=========" << std::endl;
		resetGPUMemory();
		m_terraBlockSolver->solve(d_image, d_target, d_mask, nonLinearIter, linearIter,  patchIter, weightFit, weightReg );
		copyResultToCPU();
#endif
#if RUN_TERRA_BLOCK
		std::cout << "\n\n======OPT_BLOCK=========" << std::endl;
		resetGPUMemory();
		m_optBlockSolver->solve(d_image, d_target, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		copyResultToCPU();
#endif


		return &m_result;
	}

	void copyResultToCPU() {
		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		cutilSafeCall(cudaMemcpy(m_result.getData(), d_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
	}

private:

	ColorImageR32G32B32A32 m_image;
	ColorImageR32G32B32A32 m_image1;
	ColorImageR32		   m_imageMask;

	ColorImageR32G32B32A32 m_result;

	float4*	d_image;
	float4* d_target;
	float*  d_mask;
	

	CUDAWarpingSolver*	    m_warpingSolver;
	TerraSolverPoissonImageEditing* m_terraSolver;
	TerraSolverPoissonImageEditing* m_optSolver;

	CUDAPatchSolverWarping* m_warpingSolverPatch;
	TerraSolverPoissonImageEditing* m_terraBlockSolver;
	TerraSolverPoissonImageEditing* m_optBlockSolver;
};

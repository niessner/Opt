#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"

#include "OptImageSolver.h"
#include "SolverInput.h"

class ImageSolver {
private:

    shared_ptr<SimpleBuffer>    m_result;
    SolverInput                 m_solverInput;

    CUDAWarpingSolver*  m_cudaSolver;
    OptImageSolver*	    m_optSolver;

public:
	ImageSolver(const SolverInput& input)
	{
        m_solverInput = input;

		resetGPUMemory();

		
        m_cudaSolver = new CUDAWarpingSolver(m_result->width(), m_result->height());
        m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "smoothingLaplacianFloat4AD.t", "gaussNewtonGPU");
		/*		m_terraBlockSolverFloat4 = new OptImageSolver(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianFloat4AD.t", "gaussNewtonBlockGPU");*/
		
	}

	void resetGPUMemory()
	{
        m_result = shared_ptr<SimpleBuffer>(new SimpleBuffer(m_solverInput.targetDepth, true));
	}

	~ImageSolver()
	{
		cutilSafeCall(cudaFree(d_imageFloat4));
		cutilSafeCall(cudaFree(d_targetFloat4));

		cutilSafeCall(cudaFree(d_imageFloat));
		cutilSafeCall(cudaFree(d_targetFloat));

		SAFE_DELETE(m_warpingSolver);

		SAFE_DELETE(m_terraSolverFloat4);
		//		SAFE_DELETE(m_terraBlockSolverFloat4);
	}

	ColorImageR32G32B32A32* solve()
	{
		float weightFit = 0.1f;
		float weightReg = 1.0f;
		
		unsigned int nonLinearIter = 10;
		unsigned int linearIter = 10;
		unsigned int patchIter = 16;

				
		std::cout << "CUDA" << std::endl;
		resetGPUMemory();
		m_warpingSolver->solveGN(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();
		

		std::cout << "\n\nTERRA" << std::endl;
		resetGPUMemory();
		m_terraSolverFloat4->solve(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();
		
#ifdef USE_CERES
        std::cout << "\n\nCERES" << std::endl;
        CeresSolverSmoothingLaplacianFloat4 *ceres = new CeresSolverSmoothingLaplacianFloat4();
        ceres->solve(m_image, weightFit, weightReg, m_result);
#endif


		return &m_result;
	}

	

	void copyResultToCPUFromFloat4() {
		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		cutilSafeCall(cudaMemcpy(m_result.getPointer(), d_imageFloat4, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
	}

	void copyResultToCPUFromFloat() {
		float* h_result = new float[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_result, d_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		for (unsigned int i = 0; i < m_image.getWidth()*m_image.getHeight(); i++) {
			float v = h_result[i];
			m_result.getPointer()[i] = vec4f(v,v,v, 1.0f);
		}

		delete h_result;
	}
};

#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAImageSolver.h"

#include "OptImageSolver.h"
#include "SFSSolverInput.h"

class ImageSolver {
private:

    std::shared_ptr<SimpleBuffer>   m_result;
    SFSSolverInput                  m_solverInput;

    CUDAImageSolver*  m_cudaSolver;
    OptImageSolver*	    m_optSolver;

public:
	ImageSolver(const SFSSolverInput& input)
	{
        m_solverInput = input;

		resetGPUMemory();

        m_cudaSolver = new CUDAImageSolver(m_result->width(), m_result->height());
        //m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "smoothingLaplacianFloat4AD.t", "gaussNewtonGPU");
		/*		m_terraBlockSolverFloat4 = new OptImageSolver(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianFloat4AD.t", "gaussNewtonBlockGPU");*/
		
	}

	void resetGPUMemory()
	{
        m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInput.targetDepth.get(), true));
	}

	~ImageSolver()
	{
        SAFE_DELETE(m_cudaSolver);

        SAFE_DELETE(m_optSolver);
		//		SAFE_DELETE(m_terraBlockSolverFloat4);
	}

    std::shared_ptr<SimpleBuffer> solve()
	{
		

				
		std::cout << "CUDA" << std::endl;
		resetGPUMemory();
        m_cudaSolver->solve(m_result, m_solverInput);
		

		//std::cout << "\n\nTERRA" << std::endl;
		//resetGPUMemory();
		//m_terraSolverFloat4->solve(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, patchIter, weightFit, weightReg);

		return m_result;
	}

};

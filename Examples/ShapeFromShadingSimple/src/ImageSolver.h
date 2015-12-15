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
    OptImageSolver*	  m_terraSolver;
    OptImageSolver*	  m_optSolver;

public:
	ImageSolver(const SFSSolverInput& input)
	{
        m_solverInput = input;

		resetGPUMemory();

        m_cudaSolver = new CUDAImageSolver(m_result->width(), m_result->height());
        m_terraSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShading.t", "gaussNewtonGPU");
        m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "gaussNewtonGPU");
		
		
	}

	void resetGPUMemory()
	{
        m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInput.initialUnknown.get(), true));
	}

	~ImageSolver()
	{
        SAFE_DELETE(m_cudaSolver);
        SAFE_DELETE(m_terraSolver);
        SAFE_DELETE(m_optSolver);
	}

    std::shared_ptr<SimpleBuffer> solve()
	{
				
		std::cout << "CUDA" << std::endl;
		resetGPUMemory();
        m_cudaSolver->solve(m_result, m_solverInput);
		

		std::cout << "\n\nTERRA" << std::endl;
		resetGPUMemory();
        m_terraSolver->solve(m_result, m_solverInput);

        std::cout << "\n\nOPT" << std::endl;
        resetGPUMemory();
        m_optSolver->solve(m_result, m_solverInput);

		return m_result;
	}

};

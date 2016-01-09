#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAImageSolver.h"

#include "OptImageSolver.h"
#include "CeresImageSolver.h"
#include "SFSSolverInput.h"

class ImageSolver {
private:

    std::shared_ptr<SimpleBuffer>   m_result;
    SFSSolverInput                  m_solverInput;

    CUDAImageSolver*  m_cudaSolver;
    OptImageSolver*	  m_terraSolver;
    OptImageSolver*	  m_optSolver;
    CeresImageSolver* m_ceresSolver;

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
        const bool useCUDA = true;
        const bool useTerra = true;
        const bool useOpt = true;
        const bool useCeres = false;

        if (useCUDA)
        {
            std::cout << "CUDA" << std::endl;
            resetGPUMemory();
            m_cudaSolver->solve(m_result, m_solverInput);
        }

        if (useTerra)
        {
            m_solverInput.parameters.solveCount = 0;
            std::cout << "\n\nTERRA" << std::endl;
            resetGPUMemory();
            m_terraSolver->solve(m_result, m_solverInput);
        }

        if (useOpt)
        {
            m_solverInput.parameters.solveCount = 1;
            std::cout << "\n\nOPT" << std::endl;
            resetGPUMemory();
            m_optSolver->solve(m_result, m_solverInput);
        }

#ifdef USE_CERES
        if (useCeres)
        {
            m_solverInput.parameters.solveCount = 2;
            std::cout << "\n\nCERES" << std::endl;
            m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInput.initialUnknown.get(), false));
            m_ceresSolver->solve(m_result, m_solverInput);
        }
#endif

        return m_result;
    }

};

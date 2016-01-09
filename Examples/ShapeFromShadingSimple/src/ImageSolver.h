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
    SFSSolverInput                  m_solverInputGPU;
    SFSSolverInput                  m_solverInputCPU;

    CUDAImageSolver*  m_cudaSolver;
    OptImageSolver*	  m_terraSolver;
    OptImageSolver*	  m_optSolver;
    CeresImageSolver* m_ceresSolver;

public:
    ImageSolver(const SFSSolverInput& inputGPU, const SFSSolverInput& inputCPU)
	{
        m_solverInputGPU = inputGPU;
        m_solverInputCPU = inputCPU;

		resetGPUMemory();

        m_cudaSolver = new CUDAImageSolver(m_result->width(), m_result->height());
        m_terraSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShading.t", "gaussNewtonGPU");
        m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "gaussNewtonGPU");
        m_ceresSolver = new CeresImageSolver(m_result->width(), m_result->height());
	}

	void resetGPUMemory()
	{
        m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputGPU.initialUnknown.get(), true));
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
        const bool useOpt = false;
        const bool useCeres = true;

        if (useCUDA)
        {
            std::cout << "CUDA" << std::endl;
            resetGPUMemory();
            m_cudaSolver->solve(m_result, m_solverInputGPU);
        }

        if (useTerra)
        {
            m_solverInputGPU.parameters.solveCount = 0;
            std::cout << "\n\nTERRA" << std::endl;
            resetGPUMemory();
            m_terraSolver->solve(m_result, m_solverInputGPU);
        }

        if (useOpt)
        {
            m_solverInputGPU.parameters.solveCount = 1;
            std::cout << "\n\nOPT" << std::endl;
            resetGPUMemory();
            m_optSolver->solve(m_result, m_solverInputGPU);
        }

#ifdef USE_CERES
        if (useCeres)
        {
            std::cout << "\n\nCERES" << std::endl;
            m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputCPU.initialUnknown.get(), false));
            m_ceresSolver->solve(m_result, m_solverInputCPU);
        }
#endif

        return m_result;
    }

};

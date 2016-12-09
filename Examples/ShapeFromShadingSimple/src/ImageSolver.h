#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <../../shared/cudaUtil.h>

#include "CUDAImageSolver.h"

#include "OptImageSolver.h"
#include "CeresImageSolver.h"
#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"

class ImageSolver {
private:

    std::shared_ptr<SimpleBuffer>   m_result;
    SFSSolverInput                  m_solverInputGPU;
    SFSSolverInput                  m_solverInputCPU;

    std::unique_ptr<CUDAImageSolver>  m_cudaSolver;
    std::unique_ptr<OptImageSolver>	  m_terraSolver;
    std::unique_ptr<OptImageSolver>	  m_optSolver;
    std::unique_ptr<OptImageSolver>	  m_optLMSolver;
    std::unique_ptr<CeresImageSolver> m_ceresSolver;
    std::vector<SolverIteration> m_ceresIters;
    std::vector<SolverIteration> m_optGNIters;
    std::vector<SolverIteration> m_optLMIters;

public:
    ImageSolver(const SFSSolverInput& inputGPU, const SFSSolverInput& inputCPU)
	{
        m_solverInputGPU = inputGPU;
        m_solverInputCPU = inputCPU;

		resetGPUMemory();

        m_cudaSolver = std::make_unique<CUDAImageSolver>(m_result->width(), m_result->height());
        m_terraSolver = std::make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShading.t", "gaussNewtonGPU");
        m_optSolver = std::make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "gaussNewtonGPU");
        m_optLMSolver = std::make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "LMGPU");
        //m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShadingADHacked.t", "gaussNewtonGPU");
        m_ceresSolver = std::make_unique<CeresImageSolver>(m_result->width(), m_result->height());
	}

	void resetGPUMemory()
	{
        m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputGPU.initialUnknown.get(), true));
	}

	~ImageSolver() {}

    std::shared_ptr<SimpleBuffer> solve()
    {
        const bool useCUDA = false;
        const bool useTerra = false;
        const bool useOpt = true;
        const bool useOptLM = true;
        const bool useCeres = true ;

        if (useCUDA)
        {
            std::cout << "CUDA" << std::endl;
            resetGPUMemory();
            m_cudaSolver->solve(m_result, m_solverInputGPU);
        }

        if (useTerra)
        {
            std::cout << "\n\nTERRA" << std::endl;
            resetGPUMemory();
            m_terraSolver->solve(m_result, m_solverInputGPU, m_optGNIters);
        }

        if (useOpt)
        {
            std::cout << "\n\nOPT" << std::endl;
            resetGPUMemory();
            m_optSolver->solve(m_result, m_solverInputGPU, m_optGNIters);
        }

        if (useOptLM)
        {
            std::cout << "\n\nOPT LM" << std::endl;
            resetGPUMemory();
            m_optLMSolver->solve(m_result, m_solverInputGPU, m_optLMIters);
        }

#ifdef USE_CERES
        if (useCeres)
        {
            std::cout << "\n\nCERES" << std::endl;
            m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputCPU.initialUnknown.get(), false));
            m_ceresSolver->solve(m_result, m_solverInputCPU, m_ceresIters);
        }
#endif


        std::string resultDirectory = "results/";
#   if OPT_DOUBLE_PRECISION
        std::string resultSuffix = "_double";
#   else
        std::string resultSuffix = "_float";
#   endif
        saveSolverResults(resultDirectory, resultSuffix, m_ceresIters, m_optGNIters, m_optLMIters);

        return m_result;
    }

};

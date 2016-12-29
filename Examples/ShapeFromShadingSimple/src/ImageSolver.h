#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <../../shared/cudaUtil.h>

#include "CUDAImageSolver.h"

#include "OptImageSolver.h"
#include "CeresImageSolver.h"
#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"

// From the future (C++14)
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}



class ImageSolver {
private:
    CombinedSolverParameters m_params;
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
    std::vector<SolverIteration> m_optTerraIters;

public:
    ImageSolver(const SFSSolverInput& inputGPU, const SFSSolverInput& inputCPU, bool performanceRun)
	{
        if (performanceRun) {
            m_params.useCUDA = false;
            m_params.useTerra = false;
            m_params.useOpt = true;
            m_params.useOptLM = true;
            m_params.useCeres = true;
        }

        m_solverInputGPU = inputGPU;
        m_solverInputCPU = inputCPU;

		resetGPUMemory();

        m_cudaSolver = make_unique<CUDAImageSolver>(m_result->width(), m_result->height());
        m_terraSolver = make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShading.t", "gaussNewtonGPU");
        m_optSolver = make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "gaussNewtonGPU");
        m_optLMSolver = make_unique<OptImageSolver>(m_result->width(), m_result->height(), "shapeFromShadingAD.t", "LMGPU");
        //m_optSolver = new OptImageSolver(m_result->width(), m_result->height(), "shapeFromShadingADHacked.t", "gaussNewtonGPU");
        m_ceresSolver = make_unique<CeresImageSolver>(m_result->width(), m_result->height());
	}

	void resetGPUMemory()
	{
        m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputGPU.initialUnknown.get(), true));
	}

	~ImageSolver() {}

    std::shared_ptr<SimpleBuffer> solve()
    {


        if (m_params.useCUDA)
        {
            std::cout << "CUDA" << std::endl;
            resetGPUMemory();
            m_cudaSolver->solve(m_result, m_solverInputGPU);
        }

        if (m_params.useTerra)
        {
            std::cout << "\n\nTERRA" << std::endl;
            resetGPUMemory();
            m_terraSolver->solve(m_result, m_solverInputGPU, m_optTerraIters);
        }

        if (m_params.useOpt)
        {
            std::cout << "\n\nOPT" << std::endl;
            resetGPUMemory();
            m_optSolver->solve(m_result, m_solverInputGPU, m_optGNIters);
        }

        if (m_params.useOptLM)
        {
            std::cout << "\n\nOPT LM" << std::endl;
            resetGPUMemory();
            m_optLMSolver->solve(m_result, m_solverInputGPU, m_optLMIters);
        }

#ifdef USE_CERES
        if (m_params.useCeres)
        {
            std::cout << "\n\nCERES" << std::endl;
            m_result = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(*m_solverInputCPU.initialUnknown.get(), false));
            m_ceresSolver->solve(m_result, m_solverInputCPU, m_ceresIters);
        }
#endif

        saveSolverResults("results/", OPT_DOUBLE_PRECISION ? "_double" : "_float", m_ceresIters, m_optGNIters, m_optLMIters);

        reportFinalCosts("Shape From Shading", m_params, m_optSolver->finalCost(), m_optLMSolver->finalCost(), m_ceresSolver->finalCost());
        
        return m_result;
    }

};

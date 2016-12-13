#pragma once

#include <cassert>

#include "cutil.h"
#include "../../shared/Precision.h"
#include "../../shared/CudaArray.h"
#include "../../shared/OptUtils.h"
extern "C" {
#include "Opt.h"
}

class OptImageSolver {

public:
	OptImageSolver(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t dims[] = { width, height };

        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~OptImageSolver()
	{
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

    struct IterStruct {
        unsigned int* nIter;
        unsigned int* lIter;
        unsigned int* pIter;
        IterStruct(unsigned int* n, unsigned int* l, unsigned int* p) : nIter(n), lIter(l), pIter(p) {}
    };


    void solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput, std::vector<SolverIteration>& iterationSummary)
	{
        std::vector<void*> images;
#if OPT_DOUBLE_PRECISION
        auto getDoubleArrayFromFloatImage = [](CudaArray<double>& doubleArray, std::shared_ptr<SimpleBuffer> floatImage) {
            std::vector<float> v;
            size_t size = floatImage->width() * floatImage->height();
            v.resize(size);
            cutilSafeCall(cudaMemcpy(v.data(), floatImage->data(), size*sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<double> vDouble;
            vDouble.resize(size);
            for (int i = 0; i < size; ++i) {
                vDouble[i] = (double)v[i];
            }
            doubleArray.update(vDouble);
        };

        CudaArray<double> resultDouble, targetDepthDouble, targetIntensityDouble, previousDepthDouble;
        getDoubleArrayFromFloatImage(resultDouble, result);
        getDoubleArrayFromFloatImage(targetDepthDouble, rawSolverInput.targetDepth);
        getDoubleArrayFromFloatImage(targetIntensityDouble, rawSolverInput.targetIntensity);
        getDoubleArrayFromFloatImage(previousDepthDouble, rawSolverInput.previousDepth);
        images.push_back(resultDouble.data());
        images.push_back(targetDepthDouble.data());
        images.push_back(targetIntensityDouble.data());
        images.push_back(previousDepthDouble.data());
#else
        images.push_back(result->data());
        images.push_back(rawSolverInput.targetDepth->data());
        images.push_back(rawSolverInput.targetIntensity->data());
        images.push_back(rawSolverInput.previousDepth->data());
#endif
        images.push_back(rawSolverInput.maskEdgeMap->data()); // row
        images.push_back(((unsigned char*)rawSolverInput.maskEdgeMap->data()) + (result->width() * result->height())); // col

        unsigned int nIter[] = { rawSolverInput.parameters.nNonLinearIterations, rawSolverInput.parameters.nLinIterations, rawSolverInput.parameters.nPatchIterations };
        IterStruct iterStruct(&nIter[0], &nIter[1], &nIter[2]);

        TerraSolverParameterPointers indirectParameters(rawSolverInput.parameters, images);

        launchProfiledSolve(m_optimizerState, m_plan, (void**)&indirectParameters, (void**)&iterStruct, iterationSummary);
       

#if OPT_DOUBLE_PRECISION
        std::vector<double> vDouble;
        size_t size = resultDouble.size();
        vDouble.resize(size);
        std::vector<float> v;
        v.resize(size);
        cutilSafeCall(cudaMemcpy(vDouble.data(), resultDouble.data(), size*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < size; ++i) {
            v[i] = (float)vDouble[i];
        }
        cutilSafeCall(cudaMemcpy(result->data(), v.data(), size*sizeof(float), cudaMemcpyHostToDevice));
#endif
	}

private:
	Opt_State*	m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;
};

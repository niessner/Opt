#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}
#include "../../shared/OptUtils.h"
#include "../../shared/CudaArray.h"
#include "../../shared/Precision.h"
#include <cuda_runtime.h>
#include <cudaUtil.h>

extern "C" void reshuffleToFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);
extern "C" void reshuffleFromFloat3CUDA(float2* d_x, float* d_a, float3* d_unknown, unsigned int width, unsigned int height);

class TerraSolverWarping {

public:
	TerraSolverWarping(unsigned int width, unsigned int height, unsigned int depth, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		uint32_t dims[] = { width, height, depth };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		m_width = width;
		m_height = height;
		m_depth = depth;

		CUDA_SAFE_CALL(cudaMalloc(&d_unknown, sizeof(float3)*width*height*depth));

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
		assert(d_unknown);
	}

	~TerraSolverWarping()
	{
		CUDA_SAFE_CALL(cudaFree(d_unknown));


		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}


	void solve(float3* d_x, float3* d_a, float3* d_urshape, float3* d_constraints, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg, std::vector<SolverIteration>& iters)
	{
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };
		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		

        std::vector<void*> problemParams;
        int scalarCount = m_height*m_width*m_depth * 3;
        CudaArray<double> d_xDouble3, d_aDouble3, d_urshapeDouble3, d_constraintsDouble3;
        if (OPT_DOUBLE_PRECISION) {
            auto getDoubleArrayFromFloatDevicePointer = [](CudaArray<double>& doubleArray, float* d_ptr, int size) {
                std::vector<float> v;
                v.resize(size);
                cutilSafeCall(cudaMemcpy(v.data(), d_ptr, size*sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<double> vDouble;
                vDouble.resize(size);
                for (int i = 0; i < size; ++i) {
                    vDouble[i] = (double)v[i];
                }
                doubleArray.update(vDouble);
            };

            
            getDoubleArrayFromFloatDevicePointer(d_xDouble3, (float*)d_x, scalarCount);
            getDoubleArrayFromFloatDevicePointer(d_aDouble3, (float*)d_a, scalarCount);
            getDoubleArrayFromFloatDevicePointer(d_urshapeDouble3, (float*)d_urshape, scalarCount);
            getDoubleArrayFromFloatDevicePointer(d_constraintsDouble3, (float*)d_constraints, scalarCount);


            problemParams = { d_xDouble3.data(), d_aDouble3.data(), d_urshapeDouble3.data(), d_constraintsDouble3.data(), &weightFitSqrt, &weightRegSqrt };
        }
        else {
            problemParams = { d_x, d_a, d_urshape, d_constraints, &weightFitSqrt, &weightRegSqrt };
        }

		

        launchProfiledSolve(m_optimizerState, m_plan, problemParams.data(), solverParams, iters);
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);

        if (OPT_DOUBLE_PRECISION) {
            std::vector<double> vDouble;
            vDouble.resize(scalarCount);
            std::vector<float> v;
            v.resize(scalarCount);

            cutilSafeCall(cudaMemcpy(vDouble.data(), d_xDouble3.data(), scalarCount*sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < scalarCount; ++i) {
                v[i] = (float)vDouble[i];
            }
            cutilSafeCall(cudaMemcpy(d_x, v.data(), scalarCount*sizeof(float), cudaMemcpyHostToDevice));

            cutilSafeCall(cudaMemcpy(vDouble.data(), d_aDouble3.data(), scalarCount*sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < scalarCount; ++i) {
                v[i] = (float)vDouble[i];
            }
            cutilSafeCall(cudaMemcpy(d_a, v.data(), scalarCount*sizeof(float), cudaMemcpyHostToDevice));
        }
	}

    double finalCost() const {
        return m_finalCost;
    }


    double m_finalCost = nan(nullptr);
	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;

	float3*	d_unknown;
	unsigned int m_width, m_height, m_depth;
};

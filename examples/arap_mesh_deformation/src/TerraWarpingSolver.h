#pragma once

#include <cassert>

#include "cutil.h"
#include "../../shared/OptUtils.h"
#include "../../shared/CudaArray.h"
#include "../../shared/Precision.h"
extern "C" {
#include "Opt.h"
}


template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
	type* d_ptr;
	cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

	cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
	return d_ptr;
}

class TerraWarpingSolver {

	int* d_headX;

	int* d_tailX;

	int edgeCount;

public:
	TerraWarpingSolver(unsigned int vertexCount, unsigned int E, const int* d_xCoords, const int* d_offsets, const std::string& terraFile, const std::string& optName) : 
		m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		edgeCount = (int)E;
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		int* h_offsets = (int*)malloc(sizeof(int)*(vertexCount + 1));
		cutilSafeCall(cudaMemcpy(h_offsets, d_offsets, sizeof(int)*(vertexCount + 1), cudaMemcpyDeviceToHost));

		int* h_xCoords = (int*)malloc(sizeof(int)*(edgeCount + 1));
		cutilSafeCall(cudaMemcpy(h_xCoords, d_xCoords, sizeof(int)*(edgeCount), cudaMemcpyDeviceToHost));
		h_xCoords[edgeCount] = vertexCount;

		// Convert to our edge format
		std::vector<int> h_headX;
		std::vector<int> h_tailX;
		for (int headX = 0; headX < (int)vertexCount; ++headX) {
			for (int j = h_offsets[headX]; j < h_offsets[headX + 1]; ++j) {
				h_headX.push_back(headX);
				h_tailX.push_back(h_xCoords[j]);
			}
		}

		d_headX = createDeviceBuffer(h_headX);
		d_tailX = createDeviceBuffer(h_tailX);

		uint32_t dims[] = { vertexCount };

		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);


		m_numUnknown = vertexCount;
	}

	~TerraWarpingSolver()
	{
		cutilSafeCall(cudaFree(d_headX));
		cutilSafeCall(cudaFree(d_tailX));

		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	//void solve(float3* d_unknown, float3* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
    void solveGN(
        float3* d_vertexPosFloat3,
        float3* d_anglesFloat3,
        float3* d_vertexPosFloat3Urshape,
        //int* d_numNeighbours,
        //int* d_neighbourIdx,
        //int* d_neighbourOffset,
        float3* d_vertexPosTargetFloat3,
        unsigned int nNonLinearIterations,
        unsigned int nLinearIterations,
        float weightFit,
        float weightReg,
        std::vector<SolverIteration>& iters)
    {
        unsigned int nBlockIterations = 1;	//invalid just as a dummy;

        void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

        float weightFitSqrt = sqrt(weightFit);
        float weightRegSqrt = sqrt(weightReg);

        int * d_zeros = d_headY;
        std::vector<void*> problemParams;
        CudaArray<double> d_vertexPosDouble3, d_anglesDouble3, d_vertexPosDouble3Urshape, d_vertexPosTargetDouble3;
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


            getDoubleArrayFromFloatDevicePointer(d_vertexPosDouble3, (float*)d_vertexPosFloat3, m_numUnknown * 3);
            getDoubleArrayFromFloatDevicePointer(d_anglesDouble3, (float*)d_anglesFloat3, m_numUnknown * 3);
            getDoubleArrayFromFloatDevicePointer(d_vertexPosDouble3Urshape, (float*)d_vertexPosFloat3Urshape, m_numUnknown * 3);
            getDoubleArrayFromFloatDevicePointer(d_vertexPosTargetDouble3, (float*)d_vertexPosTargetFloat3, m_numUnknown * 3);


            problemParams = { &weightFitSqrt, &weightRegSqrt, d_vertexPosDouble3.data(), d_anglesDouble3.data(), d_vertexPosDouble3Urshape.data(), d_vertexPosTargetDouble3.data(), &edgeCount, d_headX, d_headY, d_tailX, d_tailY };
        } else {
            problemParams = { &weightFitSqrt, &weightRegSqrt, d_vertexPosFloat3, d_anglesFloat3, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, &edgeCount, d_headX, d_headY, d_tailX, d_tailY };
        }
        launchProfiledSolve(m_optimizerState, m_plan, problemParams.data(), solverParams, iters);
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);

        if (OPT_DOUBLE_PRECISION) {
            std::vector<double> vDouble;
            size_t size = m_numUnknown * 3;
            vDouble.resize(size);
            std::vector<float> v;
            v.resize(size);

            cutilSafeCall(cudaMemcpy(vDouble.data(), d_vertexPosDouble3.data(), size*sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < size; ++i) {
                v[i] = (float)vDouble[i];
            }
            cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, v.data(), size*sizeof(float), cudaMemcpyHostToDevice));

            cutilSafeCall(cudaMemcpy(vDouble.data(), d_anglesDouble3.data(), size*sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < size; ++i) {
                v[i] = (float)vDouble[i];
            }
            cutilSafeCall(cudaMemcpy(d_anglesFloat3, v.data(), size*sizeof(float), cudaMemcpyHostToDevice));
        }

	}

    double finalCost() const {
        return m_finalCost;
    }

private:
	Opt_State*	m_optimizerState;
    Opt_Problem*	m_problem;
    Opt_Plan*		m_plan;

    double m_finalCost = nan(nullptr);
	unsigned int m_numUnknown;
};

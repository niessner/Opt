#pragma once

#include <cassert>

#include "cutil.h"

extern "C" {
#include "Opt.h"
}

template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
	type* d_ptr;
	cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

	cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
	return d_ptr;
}

class TerraSolverWarpingFloat4 {

	int* d_headX;
	int* d_headY;
	int* d_tailX;
	int* d_tailY;
	int edgeCount;

public:
	TerraSolverWarpingFloat4(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);


		std::vector<int> headX;
		std::vector<int> headY;
		std::vector<int> tailX;
		std::vector<int> tailY;

		for (int y = 0; y < (int)height; ++y) {
			for (int x = 0; x < (int)width; ++x) {
				if (x > 0) {
					headX.push_back(x);
					headY.push_back(y);

					tailX.push_back(x - 1);
					tailY.push_back(y);
				}
				if (y > 0) {
					headX.push_back(x);
					headY.push_back(y);

					tailX.push_back(x);
					tailY.push_back(y - 1);
				}
				if (x < (int)width - 1) {
					headX.push_back(x);
					headY.push_back(y);

					tailX.push_back(x + 1);
					tailY.push_back(y);
				}
				if (y < (int)height - 1) {
					headX.push_back(x);
					headY.push_back(y);

					tailX.push_back(x);
					tailY.push_back(y + 1);
				}
			}
		}

		edgeCount = (int)tailX.size();

		d_headX = createDeviceBuffer(headX);
		d_headY = createDeviceBuffer(headY);

		d_tailX = createDeviceBuffer(tailX);
		d_tailY = createDeviceBuffer(tailY);


		uint32_t stride = width * sizeof(float4);
		uint32_t strides[] = { stride, stride };
		uint32_t elemsizes[] = { sizeof(float4), sizeof(float4) };
		uint32_t dims[] = { width, height };

		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, elemsizes, strides);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~TerraSolverWarpingFloat4()
	{
		cutilSafeCall(cudaFree(d_headX));
		cutilSafeCall(cudaFree(d_headY));
		cutilSafeCall(cudaFree(d_tailX));
		cutilSafeCall(cudaFree(d_tailY));


		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	void solve(float4* d_unknown, float4* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		void* data[] = { d_unknown, d_target };
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		void* problemParams[] = { &weightFitSqrt, &weightRegSqrt };


		//Opt_ProblemInit(m_optimizerState, m_plan, data, NULL, problemParams, (void**)&solverParams);
		//while (Opt_ProblemStep(m_optimizerState, m_plan, data, NULL, problemParams, NULL));
		int32_t* xCoords[] = { d_headX, d_tailX };
		int32_t* yCoords[] = { d_headY, d_tailY };
		int32_t edgeCounts[] = { edgeCount };
		Opt_ProblemSolve(m_optimizerState, m_plan, data, edgeCounts, NULL, xCoords, yCoords, problemParams, solverParams);
	}

private:
	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;
};

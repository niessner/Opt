#pragma once

#include <cassert>

#include "cutil.h"

extern "C" {
#include "Opt.h"
}

struct float9 {
	float array[9];
};

template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
	type* d_ptr;
	cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

	cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
	return d_ptr;
}

class TerraSolver {

	int* d_headX;
	int* d_headY;

	int* d_tailX;
	int* d_tailY;

	int edgeCount;

public:
	TerraSolver(unsigned int vertexCount, unsigned int E, const int* d_xCoords, const int* d_offsets, const std::string& terraFile, const std::string& optName) : 
		m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		edgeCount = (int)E;
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());

		std::vector<int> yCoords;

		for (int y = 0; y < (int)edgeCount; ++y) {
			yCoords.push_back(0);
		}

		d_headY = createDeviceBuffer(yCoords);
		d_tailY = createDeviceBuffer(yCoords);

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



		uint32_t dims[] = { vertexCount, 1 };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);


		m_numUnknown = vertexCount;
	}

	~TerraSolver()
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

	//void solve(float3* d_unknown, float3* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	void solveGN(
		float3* d_vertexPosFloat3,
		float9* d_rotsFloat9,
		float3* d_vertexPosFloat3Urshape,
		float3* d_vertexPosTargetFloat3,
		unsigned int nNonLinearIterations,
		unsigned int nLinearIterations,
		float weightFit,
		float weightReg,
		float weightRot)
	{
		unsigned int nBlockIterations = 1;	//invalid just as a dummy;


		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		float weightRotSqrt = sqrt(weightRot);
        void* problemParams[] = { &weightFitSqrt, &weightRegSqrt, &weightRotSqrt, d_vertexPosFloat3, d_rotsFloat9, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, &edgeCount, d_headX, d_headY, d_tailX, d_tailY };
		Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);

        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);

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

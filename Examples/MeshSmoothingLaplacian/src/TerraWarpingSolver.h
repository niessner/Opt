#pragma once

#include <cassert>

#include "cutil.h"

extern "C" {
#include "../Opt.h"
}

template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
  type* d_ptr;
  cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

  cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
  return d_ptr;
}

class TerraWarpingSolver {

  int* d_yCoords;
  
public:
 TerraWarpingSolver(unsigned int vertexCount, unsigned int edgeCount, int* d_xCoords, int* d_offsets, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);

		std::vector<int> yCoords;

		
		for (unsigned int y = 0; y < edgeCount; ++y) {
		  yCoords.push_back(0);
		}

		d_yCoords = createDeviceBuffer(yCoords);
		
		
		uint32_t stride = vertexCount * sizeof(float3);
		uint32_t strides[] = { stride, stride };
		uint32_t elemsizes[] = { sizeof(float3), sizeof(float3) };
		uint32_t dims[] = { vertexCount, 1 };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, elemsizes, strides, &d_offsets, &d_xCoords, &d_yCoords);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~TerraWarpingSolver()
	{
	  cutilSafeCall(cudaFree(d_yCoords));

	  if (m_plan) {
	    Opt_PlanFree(m_optimizerState, m_plan);
	  }
	  
	  if (m_problem) {
	    Opt_ProblemDelete(m_optimizerState, m_problem);
	  }

	}

	void solve(float3* d_unknown, float3* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
	{

		void* data[] = {d_unknown, d_target};
		void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

		float weightFitSqrt = sqrt(weightFit);
		float weightRegSqrt = sqrt(weightReg);
		void* problemParams[] = { &weightFitSqrt, &weightRegSqrt };


		//Opt_ProblemInit(m_optimizerState, m_plan, data, NULL, problemParams, (void**)&solverParams);
		//while (Opt_ProblemStep(m_optimizerState, m_plan, data, NULL, problemParams, NULL));
		Opt_ProblemSolve(m_optimizerState, m_plan, data, NULL, problemParams, solverParams);
	}

private:
	OptState*	m_optimizerState;
	Problem*	m_problem;
	Plan*		m_plan;
};

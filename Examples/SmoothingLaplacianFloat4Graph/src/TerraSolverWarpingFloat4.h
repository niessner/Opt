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

  int* d_offsets;
  int* d_xCoords;
  int* d_yCoords;
  
public:
	TerraSolverWarpingFloat4(unsigned int width, unsigned int height, const std::string& terraFile, const std::string& optName) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
	{
		m_optimizerState = Opt_NewState();
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str(), NULL);

		std::vector<int> offsets;
		std::vector<int> xCoords;
		std::vector<int> yCoords;

		
		for (int y = 0; y < (int)height; ++y) {
		  for (int x = 0; x < (int)width; ++x) {
		    offsets.push_back((int)xCoords.size());
		    if (x > 0) {
		      xCoords.push_back(x-1);
		      yCoords.push_back(y);
		    }
		    if (y > 0) {
		      xCoords.push_back(x);
		      yCoords.push_back(y-1);
		    }
		    if (x < (int)width-1) {
		      xCoords.push_back(x+1);
		      yCoords.push_back(y);
		    }
		    if (y < (int)height-1) {
		      xCoords.push_back(x);
		      yCoords.push_back(y+1);
		    }
		  }
		}
		offsets.push_back((int)xCoords.size());
		
		d_offsets = createDeviceBuffer(offsets);
		d_xCoords = createDeviceBuffer(xCoords);
		d_yCoords = createDeviceBuffer(yCoords);
		
		
		uint32_t stride = width *sizeof(float4);
		uint32_t strides[] = { stride, stride };
		uint32_t elemsizes[] = { sizeof(float4), sizeof(float4) };
		uint32_t dims[] = { width, height };
		m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, elemsizes, strides, &d_offsets, &d_xCoords, &d_yCoords);

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

	~TerraSolverWarpingFloat4()
	{
	  cutilSafeCall(cudaFree(d_offsets));
	  cutilSafeCall(cudaFree(d_xCoords));
	  cutilSafeCall(cudaFree(d_yCoords));

	        if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}

	}

	void solve(float4* d_unknown, float4* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nBlockIterations, float weightFit, float weightReg)
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

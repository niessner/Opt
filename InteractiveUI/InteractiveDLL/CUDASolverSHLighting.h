#pragma once

#ifndef _SOLVER_SH_LIGHTING_
#define _SOLVER_SH_LIGHTING_

#include <cuda_runtime.h>

#include "SolverSHLightingState.h"

#define LE_THREAD_SIZE 16

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif
class CUDASolverSHLighting
{
public:

	CUDASolverSHLighting(int width, int height);
	~CUDASolverSHLighting();

	
	void solveLighting(SolverSHInput &input, float4* d_normalMap, float thres_depth);
	
	void solveReflectance(SolverSHInput &input, float4* d_normalMap);

private:

	float* d_litestcashe;
	float* h_litestmat;

	float* d_albedoestcashe;
	float* h_albedoestmat;
};

#endif

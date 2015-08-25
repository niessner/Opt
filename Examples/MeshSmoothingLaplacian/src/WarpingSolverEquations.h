#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"

////////////////////////////////////////
// evalF
////////////////////////////////////////

__inline__ __device__ float evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float3 e = make_float3(0.0f, 0.0f, 0.0F);

	// E_fit
	float3 targetDepth = state.d_target[variableIdx];
	float3 e_fit = (state.d_x[variableIdx] - targetDepth);
	e += parameters.weightFitting * e_fit * e_fit;

	// E_reg
	float3 p = state.d_x[variableIdx];
	int numNeighbours = input.d_numNeighbours[variableIdx];

	float3 e_reg = make_float3(0.0f, 0.0f, 0.0F);
	for (unsigned int i = 0; i < numNeighbours; i++)
	{
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		float3 q = state.d_x[neighbourIndex]; float3 v = (p - q); e_reg += make_float3(v.x*v.x, v.y*v.y, v.z*v.z);
	}
	
	e += parameters.weightRegularizer*e_reg;
	float res = e.x + e.y + e.z;

	return res;
}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float3 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	state.d_delta[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);

	float3 b   = make_float3(0.0f, 0.0f, 0.0f);
	float3 pre = make_float3(0.0f, 0.0f, 0.0f);

	// fit/pos
	b += -2.0f*parameters.weightFitting*(state.d_x[variableIdx] - state.d_target[variableIdx]);
	pre += 2.0f*parameters.weightFitting;

	// reg/pos
	float3 p = state.d_x[variableIdx];
	float3 e_reg = make_float3(0.0f, 0.0f, 0.0f);

	int numNeighbours = input.d_numNeighbours[variableIdx];
	for (unsigned int i = 0; i < numNeighbours; i++)
	{
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		e_reg += 2.0f*(p - state.d_x[neighbourIndex]); pre += 4.0f*parameters.weightRegularizer*make_float3(1.0f, 1.0f, 1.0);
	}
	b += -2.0f*parameters.weightRegularizer*e_reg;

	pre = make_float3(1.0f, 1.0f, 1.0f); // TODO!!!

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else					   pre = make_float3(1.0f, 1.0f, 1.0f);
	state.d_precondioner[variableIdx] = pre;
	
	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float3 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float3 b = make_float3(0.0f, 0.0f, 0.0f);
	float3 p = state.d_p[variableIdx];

	// fit/pos
	b += 2.0f*parameters.weightFitting*state.d_p[variableIdx];

	// pos/reg
	float3 e_reg = make_float3(0.0f, 0.0f, 0.0f);

	int numNeighbours = input.d_numNeighbours[variableIdx];
	for (unsigned int i = 0; i < numNeighbours; i++)
	{
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		e_reg += 2.0f*(p - state.d_p[neighbourIndex]);
	}
	b += 2.0f*parameters.weightRegularizer*e_reg;

	return b;
}

#endif

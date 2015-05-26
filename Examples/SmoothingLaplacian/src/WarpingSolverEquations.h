#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float4 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	state.d_delta[variableIdx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 b   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pre = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	// fit/pos
	b += -parameters.weightFitting*(state.d_x[variableIdx] - state.d_target[variableIdx]);
	pre += parameters.weightFitting;

	// reg/pos
	float4 p = state.d_x[get1DIdx(i, j, input.width, input.height)];
	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0){ float4 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; e_reg += (p - q); pre += parameters.weightRegularizer*make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN1){ float4 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; e_reg += (p - q); pre += parameters.weightRegularizer*make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN2){ float4 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; e_reg += (p - q); pre += parameters.weightRegularizer*make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN3){ float4 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; e_reg += (p - q); pre += parameters.weightRegularizer*make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	b += -parameters.weightRegularizer*e_reg;

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else					   pre = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	state.d_precondioner[variableIdx] = pre;
	
	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float4 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float4 p = state.d_p[get1DIdx(i, j, input.width, input.height)];

	// fit/pos
	b += parameters.weightFitting*state.d_p[variableIdx];

	// pos/reg
	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += (p - state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) e_reg += (p - state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) e_reg += (p - state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) e_reg += (p - state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	b += parameters.weightRegularizer*e_reg;

	return b;
}

#endif

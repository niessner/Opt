#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"

__inline__ __device__ float4 evalLaplacian(unsigned int i, unsigned int j, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	if (!isInsideImage(i, j, input.width, input.height)) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)];
	if (validN1) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)];
	if (validN2) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)];
	if (validN3) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)];

	return e_reg;
}

__inline__ __device__ float4 evalLaplacianP(unsigned int i, unsigned int j, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	if (!isInsideImage(i, j, input.width, input.height)) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += state.d_p[get1DIdx(i, j, input.width, input.height)] - state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)];
	if (validN1) e_reg += state.d_p[get1DIdx(i, j, input.width, input.height)] - state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)];
	if (validN2) e_reg += state.d_p[get1DIdx(i, j, input.width, input.height)] - state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)];
	if (validN3) e_reg += state.d_p[get1DIdx(i, j, input.width, input.height)] - state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)];

	return e_reg;
}

__inline__ __device__ float getNumNeighbors(int i, int j, SolverInput& input)
{
	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float res = 0.0f;
	if (validN0) res += 1.0f;
	if (validN1) res += 1.0f;
	if (validN2) res += 1.0f;
	if (validN3) res += 1.0f;
	return res;
}

////////////////////////////////////////
// evalF
////////////////////////////////////////
__inline__ __device__ float4 evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float4 e = make_float4(0.0f, 0.0f, 0.0F, 0.0f);

	// E_fit
	float4 targetDepth = state.d_target[variableIdx];
	float4 e_fit = (state.d_x[variableIdx] - targetDepth);
	e += parameters.weightFitting * e_fit * e_fit;
	
	// E_reg
	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	float4 e_reg = evalLaplacian(i, j, input, state, parameters);
	e += parameters.weightRegularizer * e_reg * e_reg;

	return e;
}


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
	float n = getNumNeighbors(i, j, input);
	float4 l_m  = evalLaplacian(i    , j    , input, state, parameters);
	float4 l_n0 = evalLaplacian(i + 1, j + 0, input, state, parameters);
	float4 l_n1 = evalLaplacian(i - 1, j + 0, input, state, parameters);
	float4 l_n2 = evalLaplacian(i + 0, j + 1, input, state, parameters);
	float4 l_n3 = evalLaplacian(i + 0, j - 1, input, state, parameters);
	b += -parameters.weightRegularizer*(l_m*n - l_n0 - l_n1 - l_n2 - l_n3);
	pre += parameters.weightRegularizer*(n*n + n);

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

	// reg/pos
	float n = getNumNeighbors(i, j, input);
	float4 l_m =  evalLaplacianP(i, j, input, state, parameters);
	float4 l_n0 = evalLaplacianP(i + 1, j + 0, input, state, parameters);
	float4 l_n1 = evalLaplacianP(i - 1, j + 0, input, state, parameters);
	float4 l_n2 = evalLaplacianP(i + 0, j + 1, input, state, parameters);
	float4 l_n3 = evalLaplacianP(i + 0, j - 1, input, state, parameters);
	b += parameters.weightRegularizer*(l_m*n - l_n0 - l_n1 - l_n2 - l_n3);

	return b;
}

#endif

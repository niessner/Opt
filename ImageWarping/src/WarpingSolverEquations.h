#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"

__inline__ __device__ float2 evalLaplacian(unsigned int i, unsigned int j, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	if (!isInsideImage(i, j, input.width, input.height)) return make_float2(0.0f, 0.0f);

	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)];
	if (validN1) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)];
	if (validN2) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)];
	if (validN3) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)];

	return e_reg;
}

__inline__ __device__ float2 evalLaplacian(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	// E_reg
	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	return evalLaplacian((unsigned int)i, (unsigned int)j, input, state, parameters);
}

__inline__ __device__ float getNumNeighbors(int i, int j, SolverInput& input) {

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

__inline__ __device__ float2 evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float2 e = make_float2(0.0f, 0.0f);

	// E_fit
	float2 targetUV = input.d_constraints[variableIdx]; bool validTarget = (targetUV.x >= 0 && targetUV.y >= 0);
	if (validTarget) {
		float2 e_fit = (state.d_x[variableIdx] - targetUV);
		e += (parameters.weightFitting) * e_fit * e_fit;
	}

	// E_reg
	float2 e_reg = evalLaplacian(variableIdx, input, state, parameters);
	e += (parameters.weightRegularizer) * e_reg * e_reg;

	return e;
}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float2 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float2 b = make_float2(0.0f, 0.0f);

	// Reset linearized update vector
	state.d_delta[variableIdx] = make_float2(0.0f, 0.0f);

	// E_fit
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	float2 targetUV = input.d_constraints[variableIdx]; bool validTarget = (targetUV.x >= 0 && targetUV.y >= 0);
	if (validTarget) {
		b += -parameters.weightFitting*(state.d_x[variableIdx] - targetUV);
	}

	// E_reg
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);

	float2 l = evalLaplacian(variableIdx, input, state, parameters);
	float  n = getNumNeighbors(i, j, input);

	b += -parameters.weightRegularizer*l;

	// Preconditioner depends on last solution P(input.d_x)
	float2 p = make_float2(0.0f, 0.0f);

	if(validTarget) p.x += parameters.weightFitting;
	p.x += parameters.weightRegularizer*n;

	if(validTarget) p.y += parameters.weightFitting;
	p.y += parameters.weightRegularizer*n;

	if (p.x > FLOAT_EPSILON) p.x = 1.0f / p.x;
	else					 p.x = 1.0f;

	if (p.y > FLOAT_EPSILON) p.y = 1.0f / p.y;
	else					 p.y = 1.0f;

	state.d_precondioner[variableIdx] = p;
	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float2 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float2 b = make_float2(0.0f, 0.0f);

	// E_fit
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	float2 targetUV = input.d_constraints[variableIdx]; bool validTarget = (targetUV.x >= 0 && targetUV.y >= 0);
	if (validTarget) {
		b += parameters.weightFitting*state.d_p[variableIdx];
	}

	// E_reg
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);

	float2 e_reg = make_float2(0.0f, 0.0f);
	float  n = getNumNeighbors(i, j, input);

	e_reg += n*state.d_p[variableIdx];	//diagonal of A

	// direct neighbors
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	if (validN0) e_reg += -(state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) e_reg += -(state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) e_reg += -(state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) e_reg += -(state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	
	b += parameters.weightRegularizer*e_reg;

	return b;
}

#endif

#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"



__inline__ __device__ float evalLaplacian(unsigned int i, unsigned int j, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	if (!isInsideImage(i, j, input.width, input.height)) return 0.0f;

	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
	const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
	const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
	const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

	float e_reg = 0.0f;
	if (validN0) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)];
	if (validN1) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)];
	if (validN2) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)];
	if (validN3) e_reg += state.d_x[get1DIdx(i, j, input.width, input.height)] - state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)];

	return e_reg;
}

__inline__ __device__ float evalLaplacian(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	// E_reg
	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	return evalLaplacian((unsigned int)i, (unsigned int)j, input, state, parameters);
}

__inline__ __device__ float getNumNeighbors(int i, int j, SolverInput& input) {

	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
	const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
	const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
	const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

	float res = 0.0f;
	if (validN0) res += 1.0f;
	if (validN1) res += 1.0f;
	if (validN2) res += 1.0f;
	if (validN3) res += 1.0f;
	return res;
}

__inline__ __device__ float evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float e = 0.0f;

	// E_fit
	float targetDepth = input.d_targetDepth[variableIdx]; bool validTarget = (targetDepth != MINF);
	if (validTarget) {
		float e_fit = (state.d_x[variableIdx] - input.d_targetDepth[variableIdx]);	e_fit = e_fit * e_fit;
		e += (parameters.weightFitting) * e_fit;
	}

	// E_reg
	float e_reg = evalLaplacian(variableIdx, input, state, parameters);
	e += (parameters.weightRegularizer) * e_reg * e_reg;

	return e;
}


////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////


__inline__ __device__ float evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float b = 0.0f;

	// Reset linearized update vector
	state.d_delta[variableIdx] = 0.0f;

	// E_fit
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	float targetDepth = input.d_targetDepth[variableIdx]; bool validTarget = (targetDepth != MINF);
	if (validTarget) {
		b += -2.0f*parameters.weightFitting*(state.d_x[variableIdx] - input.d_targetDepth[variableIdx]);
	}

	// E_reg
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);

	float l_m = evalLaplacian(variableIdx, input, state, parameters);
	float l_n0 = evalLaplacian(i + 1, j + 0, input, state, parameters);
	float l_n1 = evalLaplacian(i - 1, j + 0, input, state, parameters);
	float l_n2 = evalLaplacian(i + 0, j + 1, input, state, parameters);
	float l_n3 = evalLaplacian(i + 0, j - 1, input, state, parameters);

	float n = getNumNeighbors(i, j, input);

	b += -parameters.weightRegularizer*2.0f*(l_m*n - l_n0 - l_n1 - l_n2 - l_n3);

	// Preconditioner depends on last solution P(input.d_x)
	float p = 0.0f;

	if (validTarget) p += 2.0f*parameters.weightFitting;	//e_reg
	p += 2.0f*parameters.weightRegularizer*(n*n + n);


	if (p > FLOAT_EPSILON)	state.d_precondioner[variableIdx] = 1.0f / p;
	else					state.d_precondioner[variableIdx] = 1.0f;



	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////


__inline__ __device__ float applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float b = 0.0f;

	// E_fit
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	float targetDepth = input.d_targetDepth[variableIdx]; bool validTarget = (targetDepth != MINF);
	if (validTarget) {
		b += 2.0f*parameters.weightFitting*state.d_p[variableIdx];
	}

	// E_reg
	// J depends on last solution J(input.d_x) and multiplies it with d_Jp returns result

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);

	float e_reg = 0.0f;
	float n = getNumNeighbors(i, j, input);

	e_reg += (n*n + n) * state.d_p[variableIdx];	//diagonal of A

	{
		// direct neighbors
		const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
		const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
		const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
		const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

		if (validN0) e_reg += -(n + getNumNeighbors(n0_i, n0_j, input))*(state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
		if (validN1) e_reg += -(n + getNumNeighbors(n1_i, n1_j, input))*(state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
		if (validN2) e_reg += -(n + getNumNeighbors(n2_i, n2_j, input))*(state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
		if (validN3) e_reg += -(n + getNumNeighbors(n3_i, n3_j, input))*(state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	}
	{
		// neighbors of neighbors
		const int n0_i = i;		const int n0_j = j - 2; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
		const int n1_i = i;		const int n1_j = j + 2; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
		const int n2_i = i - 2; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
		const int n3_i = i + 2; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

		if (validN0) e_reg += 1.0f*(state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
		if (validN1) e_reg += 1.0f*(state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
		if (validN2) e_reg += 1.0f*(state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
		if (validN3) e_reg += 1.0f*(state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	}
	{
		// diagonal neighbors
		const int n0_i = i + 1;	const int n0_j = j + 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
		const int n1_i = i + 1;	const int n1_j = j - 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
		const int n2_i = i - 1; const int n2_j = j + 1;	const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
		const int n3_i = i - 1; const int n3_j = j - 1;	const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

		if (validN0) e_reg += 2.0f*(state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
		if (validN1) e_reg += 2.0f*(state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
		if (validN2) e_reg += 2.0f*(state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
		if (validN3) e_reg += 2.0f*(state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	}
	
	b += 2.0f*parameters.weightRegularizer*e_reg;

	//printf("e=%.2f (%d,%d)\t",e_reg,i,j);
	return b;
}


#endif

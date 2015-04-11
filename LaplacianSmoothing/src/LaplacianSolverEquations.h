#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "LaplacianSolverUtil.h"
#include "LaplacianSolverState.h"
#include "LaplacianSolverParameters.h"

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
	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);

	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
	const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
	const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
	const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

	float e_reg = 0.0f;
	if (validN0) e_reg += state.d_x[variableIdx] - state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)];
	if (validN1) e_reg += state.d_x[variableIdx] - state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)];
	if (validN2) e_reg += state.d_x[variableIdx] - state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)];
	if (validN3) e_reg += state.d_x[variableIdx] - state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)];

	e_reg = e_reg*e_reg;
	e += (parameters.weightRegularizer) * e_reg;

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

	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
	const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
	const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
	const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

	if (validN0) b += -parameters.weightRegularizer*2.0f*(state.d_x[variableIdx] - state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) b += -parameters.weightRegularizer*2.0f*(state.d_x[variableIdx] - state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) b += -parameters.weightRegularizer*2.0f*(state.d_x[variableIdx] - state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) b += -parameters.weightRegularizer*2.0f*(state.d_x[variableIdx] - state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]);

	// Preconditioner depends on last solution P(input.d_x)
	float p = 0.0f;

	if (validTarget) p += 2.0f*parameters.weightFitting;

	if (validN0) p += 2.0f*parameters.weightRegularizer;
	if (validN1) p += 2.0f*parameters.weightRegularizer;
	if (validN2) p += 2.0f*parameters.weightRegularizer;
	if (validN3) p += 2.0f*parameters.weightRegularizer;

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

	// in the case of a graph/mesh these 'neighbor' indices need to be obtained by the domain data structure (e.g., CRS/CCS)
	const int n0_i = i;		const int n0_j = j - 1; const bool isInsideN0 = isInsideImage(n0_i, n0_j, input.width, input.height); bool validN0 = false; if (isInsideN0) { float neighbourDepth0 = input.d_targetDepth[get1DIdx(n0_i, n0_j, input.width, input.height)]; validN0 = (neighbourDepth0 != MINF); }
	const int n1_i = i;		const int n1_j = j + 1; const bool isInsideN1 = isInsideImage(n1_i, n1_j, input.width, input.height); bool validN1 = false; if (isInsideN1) { float neighbourDepth1 = input.d_targetDepth[get1DIdx(n1_i, n1_j, input.width, input.height)]; validN1 = (neighbourDepth1 != MINF); }
	const int n2_i = i - 1; const int n2_j = j;		const bool isInsideN2 = isInsideImage(n2_i, n2_j, input.width, input.height); bool validN2 = false; if (isInsideN2) { float neighbourDepth2 = input.d_targetDepth[get1DIdx(n2_i, n2_j, input.width, input.height)]; validN2 = (neighbourDepth2 != MINF); }
	const int n3_i = i + 1; const int n3_j = j;		const bool isInsideN3 = isInsideImage(n3_i, n3_j, input.width, input.height); bool validN3 = false; if (isInsideN3) { float neighbourDepth3 = input.d_targetDepth[get1DIdx(n3_i, n3_j, input.width, input.height)]; validN3 = (neighbourDepth3 != MINF); }

	if (validN0) b += 2.0f*parameters.weightRegularizer*(state.d_p[variableIdx] - state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) b += 2.0f*parameters.weightRegularizer*(state.d_p[variableIdx] - state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) b += 2.0f*parameters.weightRegularizer*(state.d_p[variableIdx] - state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) b += 2.0f*parameters.weightRegularizer*(state.d_p[variableIdx] - state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);

	return b;
}

#endif

#pragma once

#ifndef _PATCH_SOLVER_Stereo_EQUATIONS_
#define _PATCH_SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "PatchSolverWarpingUtil.h"
#include "PatchSolverWarpingState.h"
#include "PatchSolverWarpingParameters.h"

__inline__ __device__ float4 evalLaplacianP(unsigned int tId_i, unsigned int tId_j, volatile float4* cacheX, volatile float4* cache, unsigned int W, unsigned int H)
{
	float4 X_CC = readValueFromCache2D(cacheX, tId_i, tId_j);

	if (!isValid(X_CC)) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 X_CM = readValueFromCache2D(cacheX, tId_i, tId_j - 1);
	float4 X_CP = readValueFromCache2D(cacheX, tId_i, tId_j + 1);
	float4 X_MC = readValueFromCache2D(cacheX, tId_i - 1, tId_j);
	float4 X_PC = readValueFromCache2D(cacheX, tId_i + 1, tId_j);

	float4 P_CC = readValueFromCache2D(cache, tId_i, tId_j);
	float4 P_CM = readValueFromCache2D(cache, tId_i, tId_j - 1);
	float4 P_CP = readValueFromCache2D(cache, tId_i, tId_j + 1);
	float4 P_MC = readValueFromCache2D(cache, tId_i - 1, tId_j);
	float4 P_PC = readValueFromCache2D(cache, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM);
	const bool validN1 = isValid(X_CP);
	const bool validN2 = isValid(X_MC);
	const bool validN3 = isValid(X_PC);

	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += P_CC - P_CM;
	if (validN1) e_reg += P_CC - P_CP;
	if (validN2) e_reg += P_CC - P_MC;
	if (validN3) e_reg += P_CC - P_PC;

	return e_reg;
}

__inline__ __device__ float4 evalLaplacian(unsigned int tId_i, unsigned int tId_j, volatile float4* cache, unsigned int W, unsigned int H)
{
	float4 X_CC = readValueFromCache2D(cache, tId_i, tId_j);

	if (!isValid(X_CC)) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 X_CM = readValueFromCache2D(cache, tId_i, tId_j - 1);
	float4 X_CP = readValueFromCache2D(cache, tId_i, tId_j + 1);
	float4 X_MC = readValueFromCache2D(cache, tId_i - 1, tId_j);
	float4 X_PC = readValueFromCache2D(cache, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM);
	const bool validN1 = isValid(X_CP);
	const bool validN2 = isValid(X_MC);
	const bool validN3 = isValid(X_PC);

	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += X_CC - X_CM;
	if (validN1) e_reg += X_CC - X_CP;
	if (validN2) e_reg += X_CC - X_MC;
	if (validN3) e_reg += X_CC - X_PC;

	return e_reg;
}

__inline__ __device__ float getNumNeighbors(int tId_i, int tId_j, volatile float4* cache, unsigned int W, unsigned int H)
{
	float4 X_CM = readValueFromCache2D(cache, tId_i, tId_j - 1);
	float4 X_CP = readValueFromCache2D(cache, tId_i, tId_j + 1);
	float4 X_MC = readValueFromCache2D(cache, tId_i - 1, tId_j);
	float4 X_PC = readValueFromCache2D(cache, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM);
	const bool validN1 = isValid(X_CP);
	const bool validN2 = isValid(X_MC);
	const bool validN3 = isValid(X_PC);

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

__inline__ __device__ float4 evalLaplacian(unsigned int i, unsigned int j, PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
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

__inline__ __device__ float4 evalFDevice(unsigned int variableIdx, PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
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
// evalMinusJTF
////////////////////////////////////////

__inline__ __device__ float4 evalMinusJTFDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, volatile float4* inTarget, volatile float4* inX, PatchSolverParameters& parameters, float4& outPre)
{
	float4 b   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pre = make_float4(0.0f, 0.0f, 0.0f, 0.0f);    

	float4 X_CC = readValueFromCache2D(inX, tId_i, tId_j);

	// fit/pos
	float4 t = readValueFromCache2D(inTarget, tId_i, tId_j);
	b += -parameters.weightFitting*(X_CC - t);
	pre += parameters.weightFitting;

	// reg/pos
	float n = getNumNeighbors(tId_i, tId_j, inX, W, H);
	float4 l_m  = evalLaplacian(tId_i    , tId_j    , inX, W, H);
	float4 l_n0 = evalLaplacian(tId_i + 1, tId_j + 0, inX, W, H);
	float4 l_n1 = evalLaplacian(tId_i - 1, tId_j + 0, inX, W, H);
	float4 l_n2 = evalLaplacian(tId_i + 0, tId_j + 1, inX, W, H);
	float4 l_n3 = evalLaplacian(tId_i + 0, tId_j - 1, inX, W, H);
	
	b += -parameters.weightRegularizer*(l_m*n - l_n0 - l_n1 - l_n2 - l_n3);
	pre += parameters.weightRegularizer*(n*n + n);

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else				       pre = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	outPre = pre;

	return b;
}

////////////////////////////////////////
// applyJTJ
////////////////////////////////////////

__inline__ __device__ float4 applyJTJDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, volatile float4* inTarget, volatile float4* inP, volatile float4* inX, PatchSolverParameters& parameters)
{
	float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	float4 P_CC = readValueFromCache2D(inP, tId_i, tId_j);    

	// fit/pos
	b += parameters.weightFitting*P_CC;

	// reg/pos
	float n = getNumNeighbors(tId_i, tId_j, inX, W, H);
	float4 l_m = evalLaplacianP(tId_i	  , tId_j	 , inX, inP, W, H);
	float4 l_n0 = evalLaplacianP(tId_i + 1, tId_j + 0, inX, inP, W, H);
	float4 l_n1 = evalLaplacianP(tId_i - 1, tId_j + 0, inX, inP, W, H);
	float4 l_n2 = evalLaplacianP(tId_i + 0, tId_j + 1, inX, inP, W, H);
	float4 l_n3 = evalLaplacianP(tId_i + 0, tId_j - 1, inX, inP, W, H);
	b += parameters.weightRegularizer*(l_m*n - l_n0 - l_n1 - l_n2 - l_n3);

	return b;
}

#endif

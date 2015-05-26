#pragma once

#ifndef _PATCH_SOLVER_Stereo_EQUATIONS_
#define _PATCH_SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "PatchSolverWarpingUtil.h"
#include "PatchSolverWarpingState.h"
#include "PatchSolverWarpingParameters.h"

////////////////////////////////////////
// evalMinusJTF
////////////////////////////////////////

__inline__ __device__ float evalMinusJTFDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, volatile float* inTarget, volatile float* inX, PatchSolverParameters& parameters, float& outPre)
{
	float b   = 0.0f;
	float pre = 0.0f;

	float X_CC = readValueFromCache2D(inX, tId_i, tId_j);	     
	float X_CM = readValueFromCache2D(inX, tId_i, tId_j - 1);
	float X_CP = readValueFromCache2D(inX, tId_i, tId_j + 1);
	float X_MC = readValueFromCache2D(inX, tId_i - 1, tId_j);
	float X_PC = readValueFromCache2D(inX, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM);
	const bool validN1 = isValid(X_CP);
	const bool validN2 = isValid(X_MC);
	const bool validN3 = isValid(X_PC);

	// fit/pos
	float t = readValueFromCache2D(inTarget, tId_i, tId_j);
	b += -parameters.weightFitting*(X_CC - t);
	pre += parameters.weightFitting;

	// reg/pos
	float p = X_CC;
	float e_reg = 0.0f;
	if (validN0){ float q = X_CM; e_reg += (p - q); pre += parameters.weightRegularizer*1.0f; }
	if (validN1){ float q = X_CP; e_reg += (p - q); pre += parameters.weightRegularizer*1.0f; }
	if (validN2){ float q = X_MC; e_reg += (p - q); pre += parameters.weightRegularizer*1.0f; }
	if (validN3){ float q = X_PC; e_reg += (p - q); pre += parameters.weightRegularizer*1.0f; }
	b += -parameters.weightRegularizer*e_reg;

	// Preconditioner
	if (pre > FLOAT_EPSILON) pre = 1.0f / pre;
	else				       pre = 1.0f;
	outPre = pre;

	return b;
}

////////////////////////////////////////
// applyJTJ
////////////////////////////////////////

__inline__ __device__ float applyJTJDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, volatile float* inTarget, volatile float* inP, volatile float* inX, PatchSolverParameters& parameters)
{
	float b = 0.0f;

	float X_CC = readValueFromCache2D(inX, tId_i, tId_j);		float P_CC = readValueFromCache2D(inP, tId_i, tId_j);    
	float X_CM = readValueFromCache2D(inX, tId_i, tId_j - 1);	float P_CM = readValueFromCache2D(inP, tId_i, tId_j - 1);
	float X_CP = readValueFromCache2D(inX, tId_i, tId_j + 1);	float P_CP = readValueFromCache2D(inP, tId_i, tId_j + 1);
	float X_MC = readValueFromCache2D(inX, tId_i - 1, tId_j);	float P_MC = readValueFromCache2D(inP, tId_i - 1, tId_j);
	float X_PC = readValueFromCache2D(inX, tId_i + 1, tId_j);	float P_PC = readValueFromCache2D(inP, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM);
	const bool validN1 = isValid(X_CP);
	const bool validN2 = isValid(X_MC);
	const bool validN3 = isValid(X_PC);

	// fit/pos
	b += parameters.weightFitting*P_CC;

	// pos/reg
	float e_reg = 0.0f;
	if (validN0) e_reg += (P_CC - P_CM);
	if (validN1) e_reg += (P_CC - P_CP);
	if (validN2) e_reg += (P_CC - P_MC);
	if (validN3) e_reg += (P_CC - P_PC);
	b += parameters.weightRegularizer*e_reg;

	return b;
}

#endif

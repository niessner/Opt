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

__inline__ __device__ float2 evalMinusJTFDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, float2 constraintUV, volatile float* inMask, volatile float2* inUrshape, volatile float2* inX, volatile float* inA, PatchSolverParameters& parameters, float2& outPre, float& outPreA, float& bA)
{
	float2 b = make_float2(0.0f, 0.0f);
	bA = 0.0f;

	float2 pre = make_float2(0.0f, 0.0f);
	float preA = 0.0f;

	float2 X_CC = readValueFromCache2D(inX, tId_i, tId_j);	   float M_CC = readValueFromCache2D(inMask, tId_i, tId_j);		float2 U_CC = readValueFromCache2D(inUrshape, tId_i, tId_j);	 float A_CC = readValueFromCache2D(inA, tId_i, tId_j);
	float2 X_CM = readValueFromCache2D(inX, tId_i, tId_j - 1); float M_CM = readValueFromCache2D(inMask, tId_i, tId_j - 1); float2 U_CM = readValueFromCache2D(inUrshape, tId_i, tId_j - 1); float A_CM = readValueFromCache2D(inA, tId_i, tId_j - 1);
	float2 X_CP = readValueFromCache2D(inX, tId_i, tId_j + 1); float M_CP = readValueFromCache2D(inMask, tId_i, tId_j + 1); float2 U_CP = readValueFromCache2D(inUrshape, tId_i, tId_j + 1); float A_CP = readValueFromCache2D(inA, tId_i, tId_j + 1);
	float2 X_MC = readValueFromCache2D(inX, tId_i - 1, tId_j); float M_MC = readValueFromCache2D(inMask, tId_i - 1, tId_j); float2 U_MC = readValueFromCache2D(inUrshape, tId_i - 1, tId_j); float A_MC = readValueFromCache2D(inA, tId_i - 1, tId_j);
	float2 X_PC = readValueFromCache2D(inX, tId_i + 1, tId_j); float M_PC = readValueFromCache2D(inMask, tId_i + 1, tId_j); float2 U_PC = readValueFromCache2D(inUrshape, tId_i + 1, tId_j); float A_PC = readValueFromCache2D(inA, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM) && M_CM == 0;
	const bool validN1 = isValid(X_CP) && M_CP == 0;
	const bool validN2 = isValid(X_MC) && M_MC == 0;
	const bool validN3 = isValid(X_PC) && M_PC == 0;

	// fit/pos
	bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && M_CC == 0;
	if (validConstraint) { b += -parameters.weightFitting*(X_CC - constraintUV); pre += parameters.weightFitting*make_float2(1.0f, 1.0f); }

	// reg/pos
	float2	 p = X_CC;
	float2	 pHat = U_CC;
	float2x2 R_i = evalR(A_CC);
	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0){ float2 q = X_CM; float2 qHat = U_CM; float2x2 R_j = evalR(A_CM); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN1){ float2 q = X_CP; float2 qHat = U_CP; float2x2 R_j = evalR(A_CP); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN2){ float2 q = X_MC; float2 qHat = U_MC; float2x2 R_j = evalR(A_MC); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN3){ float2 q = X_PC; float2 qHat = U_PC; float2x2 R_j = evalR(A_PC); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	b += -parameters.weightRegularizer*e_reg;

	// reg/angle
	float2x2 R = evalR(A_CC);
	float2x2 dR = evalR_dR(A_CC);
	float e_reg_angle = 0.0f;
	if (validN0) { float2 q = X_CM; float2 qHat = U_CM; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN1) { float2 q = X_CP; float2 qHat = U_CP; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN2) { float2 q = X_MC; float2 qHat = U_MC; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN3) { float2 q = X_PC; float2 qHat = U_PC; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	bA += -parameters.weightRegularizer*e_reg_angle;

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else				       pre = make_float2(1.0f, 1.0f);
	outPre = pre;

	// Preconditioner
	if (preA > FLOAT_EPSILON) preA = 1.0f / preA;
	else					  preA = 1.0f;
	outPreA = preA;

	return b;
}

////////////////////////////////////////
// applyJTJ
////////////////////////////////////////

__inline__ __device__ float2 applyJTJDevice(int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H, float2 constraintUV, volatile float* inMask, volatile float2* inUrshape, volatile float2* inP, volatile float* inPA, volatile float2* inX, volatile float* inA, PatchSolverParameters& parameters, float& bA)
{
	float2 b = make_float2(0.0f, 0.0f);
	bA = 0.0f;

	float2 X_CC = readValueFromCache2D(inX, tId_i, tId_j);	   float M_CC = readValueFromCache2D(inMask, tId_i, tId_j);		float2 U_CC = readValueFromCache2D(inUrshape, tId_i, tId_j);	 float A_CC = readValueFromCache2D(inA, tId_i, tId_j);	   float2 P_CC = readValueFromCache2D(inP, tId_i, tId_j);     float PA_CC = readValueFromCache2D(inPA, tId_i, tId_j);
	float2 X_CM = readValueFromCache2D(inX, tId_i, tId_j - 1); float M_CM = readValueFromCache2D(inMask, tId_i, tId_j - 1); float2 U_CM = readValueFromCache2D(inUrshape, tId_i, tId_j - 1); float A_CM = readValueFromCache2D(inA, tId_i, tId_j - 1); float2 P_CM = readValueFromCache2D(inP, tId_i, tId_j - 1); float PA_CM = readValueFromCache2D(inPA, tId_i, tId_j - 1);
	float2 X_CP = readValueFromCache2D(inX, tId_i, tId_j + 1); float M_CP = readValueFromCache2D(inMask, tId_i, tId_j + 1); float2 U_CP = readValueFromCache2D(inUrshape, tId_i, tId_j + 1); float A_CP = readValueFromCache2D(inA, tId_i, tId_j + 1); float2 P_CP = readValueFromCache2D(inP, tId_i, tId_j + 1); float PA_CP = readValueFromCache2D(inPA, tId_i, tId_j + 1);
	float2 X_MC = readValueFromCache2D(inX, tId_i - 1, tId_j); float M_MC = readValueFromCache2D(inMask, tId_i - 1, tId_j); float2 U_MC = readValueFromCache2D(inUrshape, tId_i - 1, tId_j); float A_MC = readValueFromCache2D(inA, tId_i - 1, tId_j); float2 P_MC = readValueFromCache2D(inP, tId_i - 1, tId_j); float PA_MC = readValueFromCache2D(inPA, tId_i - 1, tId_j);
	float2 X_PC = readValueFromCache2D(inX, tId_i + 1, tId_j); float M_PC = readValueFromCache2D(inMask, tId_i + 1, tId_j); float2 U_PC = readValueFromCache2D(inUrshape, tId_i + 1, tId_j); float A_PC = readValueFromCache2D(inA, tId_i + 1, tId_j); float2 P_PC = readValueFromCache2D(inP, tId_i + 1, tId_j); float PA_PC = readValueFromCache2D(inPA, tId_i + 1, tId_j);

	const bool validN0 = isValid(X_CM) && M_CM == 0;
	const bool validN1 = isValid(X_CP) && M_CP == 0;
	const bool validN2 = isValid(X_MC) && M_MC == 0;
	const bool validN3 = isValid(X_PC) && M_PC == 0;

	// pos/constraint
	bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && M_CC == 0;
	if (validConstraint) { b += parameters.weightFitting*P_CC; }

	// pos/reg
	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0) e_reg += 2.0f*(P_CC - P_CM);
	if (validN1) e_reg += 2.0f*(P_CC - P_CP);
	if (validN2) e_reg += 2.0f*(P_CC - P_MC);
	if (validN3) e_reg += 2.0f*(P_CC - P_PC);
	b += parameters.weightRegularizer*e_reg;

	// angle/reg
	float	 e_reg_angle = 0.0f;
	float	 angleP		 = PA_CC;
	float2x2 dR			 = evalR_dR(PA_CC);
	float2   pHat		 = U_CC;
	if (validN0) { float2 qHat = U_CM; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN1) { float2 qHat = U_CP; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN2) { float2 qHat = U_MC; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN3) { float2 qHat = U_PC; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	bA += parameters.weightRegularizer*e_reg_angle;

	return b;
}

#endif

#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"

__inline__ __device__ float evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float2 e = make_float2(0.0f, 0.0f);

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height); if(validN0) { validN0 = (state.d_mask[get1DIdx(n0_i, n0_j, input.width, input.height)] == 0); };
	const int n1_i = i;		const int n1_j = j + 1; bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height); if(validN1) { validN1 = (state.d_mask[get1DIdx(n1_i, n1_j, input.width, input.height)] == 0); };
	const int n2_i = i - 1; const int n2_j = j;		bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height); if(validN2) { validN2 = (state.d_mask[get1DIdx(n2_i, n2_j, input.width, input.height)] == 0); };
	const int n3_i = i + 1; const int n3_j = j;		bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height); if(validN3) { validN3 = (state.d_mask[get1DIdx(n3_i, n3_j, input.width, input.height)] == 0); };

	// E_fit
	float2 constraintUV = input.d_constraints[variableIdx];	bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && state.d_mask[get1DIdx(i, j, input.width, input.height)] == 0;
	if (validConstraint) { 
		float2 e_fit = (state.d_x[variableIdx] - constraintUV); 
		e += parameters.weightFitting*e_fit*e_fit; 
		//printf("e=%f | %f (%d|%d)\n", e.x, e.y, i, j);
		//printf("x=%f | %f\n", state.d_x[variableIdx].x, state.d_x[variableIdx].y);
	}

	// E_reg
	float2x2 R = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
	float2   p = state.d_x[get1DIdx(i, j, input.width, input.height)];
	float2   pHat = state.d_urshape[get1DIdx(i, j, input.width, input.height)];
	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0) { float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
	if (validN1) { float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
	if (validN2) { float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
	if (validN3) { float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
	e += parameters.weightRegularizer*e_reg;

	float res = e.x + e.y;
	return res;
}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float2 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float& bA)
{
	state.d_delta[variableIdx] = make_float2(0.0f, 0.0f);
	state.d_deltaA[variableIdx] = 0.0f;

	float2 b = make_float2(0.0f, 0.0f);
	bA = 0.0f;

	float2 pre = make_float2(0.0f, 0.0f);
	float preA = 0.0f;

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height) && state.d_mask[get1DIdx(n0_i, n0_j, input.width, input.height)] == 0;
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height) && state.d_mask[get1DIdx(n1_i, n1_j, input.width, input.height)] == 0;
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height) && state.d_mask[get1DIdx(n2_i, n2_j, input.width, input.height)] == 0;
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height) && state.d_mask[get1DIdx(n3_i, n3_j, input.width, input.height)] == 0;

	// fit/pos
	float2 constraintUV = input.d_constraints[variableIdx];	bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && state.d_mask[get1DIdx(i, j, input.width, input.height)] == 0;
	if (validConstraint) { b += -2.0f*parameters.weightFitting*(state.d_x[variableIdx] - constraintUV); pre += 2.0f*parameters.weightFitting*make_float2(1.0f, 1.0f); }

	// reg/pos
	float2	 p = state.d_x[get1DIdx(i, j, input.width, input.height)];
	float2	 pHat = state.d_urshape[get1DIdx(i, j, input.width, input.height)];
	float2x2 R_i = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0){ float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2x2 R_j = evalR(state.d_A[get1DIdx(n0_i, n0_j, input.width, input.height)]); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN1){ float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2x2 R_j = evalR(state.d_A[get1DIdx(n1_i, n1_j, input.width, input.height)]); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN2){ float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2x2 R_j = evalR(state.d_A[get1DIdx(n2_i, n2_j, input.width, input.height)]); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	if (validN3){ float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2x2 R_j = evalR(state.d_A[get1DIdx(n3_i, n3_j, input.width, input.height)]); e_reg += 2 * (p - q) - float2(mat2x2(R_i + R_j)*mat2x1(pHat - qHat)); pre += 2.0f*parameters.weightRegularizer; }
	b += -2.0f*parameters.weightRegularizer*e_reg;

	// reg/angle
	float2x2 R = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
	float2x2 dR = evalR_dR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
	float e_reg_angle = 0.0f;
	if (validN0) { float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN1) { float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN2) { float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	if (validN3) { float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; mat2x1 D = -mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); preA += D.getTranspose()*D*parameters.weightRegularizer; }
	bA += -4.0f*parameters.weightRegularizer*e_reg_angle;

	//TODO ENABLE OUR PRE-CONDITIONER AGAIN
	pre = make_float2(1.0f, 1.0f);
	preA = 1.0f;

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else				       pre = make_float2(1.0f, 1.0f);
	state.d_precondioner[variableIdx] = pre;

	// Preconditioner
	if (preA > FLOAT_EPSILON) preA = 1.0f / preA;
	else					  preA = 1.0f;
	state.d_precondionerA[variableIdx] = preA;
	
	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float2 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float& bA)
{
	float2 b = make_float2(0.0f, 0.0f);
	bA = 0.0f;

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height) && state.d_mask[get1DIdx(n0_i, n0_j, input.width, input.height)] == 0;
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height) && state.d_mask[get1DIdx(n1_i, n1_j, input.width, input.height)] == 0;
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height) && state.d_mask[get1DIdx(n2_i, n2_j, input.width, input.height)] == 0;
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height) && state.d_mask[get1DIdx(n3_i, n3_j, input.width, input.height)] == 0;

	// pos/constraint
	float2 constraintUV = input.d_constraints[variableIdx];	bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && state.d_mask[get1DIdx(i, j, input.width, input.height)] == 0;
	if (validConstraint) { b += 2.0f*parameters.weightFitting*state.d_p[variableIdx]; }

	// pos/reg
	float2 e_reg = make_float2(0.0f, 0.0f);
	if (validN0) e_reg += 2.0f*(state.d_p[variableIdx] - state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) e_reg += 2.0f*(state.d_p[variableIdx] - state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) e_reg += 2.0f*(state.d_p[variableIdx] - state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) e_reg += 2.0f*(state.d_p[variableIdx] - state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	b += 2.0f*parameters.weightRegularizer*e_reg;

	// angle/reg
	float	 e_reg_angle = 0.0f;
	float	 angleP		 = state.d_pA[variableIdx];
	float2x2 dR			 = evalR_dR(state.d_A[variableIdx]);
	float2   pHat		 = state.d_urshape[get1DIdx(i, j, input.width, input.height)];
	if (validN0) { float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN1) { float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN2) { float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	if (validN3) { float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; mat2x1 D = mat2x1(dR*(pHat - qHat)); e_reg_angle += D.getTranspose()*D*angleP; }
	bA += 4.0f*parameters.weightRegularizer*e_reg_angle;

	return b;
}

#endif

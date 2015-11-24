#pragma once

#ifndef _SOLVER_Stereo_EQUATIONS_
#define _SOLVER_Stereo_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "SolverSFSState.h"
#include "../PatchSolverSFS/PatchSolverSFSParameters.h"
#include "../PatchSolverSFS/PatchSolverSFSState.h"
#include "../PatchSolverSFS/PatchSolverSFSUtil.h"

#include "../PatchSolverSFS/SolverTerms.h"

#define DEPTH_DISCONTINUITY_THRE 0.01


__inline__ __device__ float readX(SolverState& state, int posy, int posx, int W) {
    return state.d_x[posy*W + posx];
}


__inline__ __device__ float4 calShading2depthGrad(SolverState& state, int posx, int posy, PatchSolverInput &input)
{
    const int W = input.width;
    const float d0 = readX(state, posy, posx - 1, W);
    const float d1 = readX(state, posy, posx, W);
    const float d2 = readX(state, posy - 1, posx, W);

    //calShading2depthGradHelper(d0, d1, d2, posx, posy, input);
    return make_float4(0, 0, 0, 0);
}


////////////////////////////////////////
// evalF
////////////////////////////////////////
__inline__ __device__ float evalFDevice(int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);

    const int W = input.width;
    const int H = input.height;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;


    // Common stuff
    const float targetDepth = input.d_targetDepth[variableIdx]; const bool validTarget = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];


    float cost = 0.0f;

    float E_s = 0.0f;
    float E_p = 0.0f;
    float E_r_h = 0.0f;
    float E_r_v = 0.0f;
    float E_r_d = 0.0f;
    float E_g_v = 0.0f;
    float E_g_h = 0.0f;

    if (validTarget)
    {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){

            float sum, tmpval;
            float val0, val1, val2;
            unsigned char maskval = 1;
#           if USE_SHADING_CONSTRAINT
                float4 temp00 = calShading2depthGrad(state, posx, posy, input);
                float4 temp10 = calShading2depthGrad(state, posx+1, posy, input);
                float4 temp01 = calShading2depthGrad(state, posx, posy+1, input);

                val0 = temp00.y;
                val1 = temp10.x;
                val2 = temp01.z;
                E_g_h = (temp00.w - temp10.w);
                E_g_v = (temp00.w - temp01.w);
#               ifdef USE_MASK_REFINE
                    E_g_h *= input.d_maskEdgeMap[variableIdx];
                    E_g_v *= input.d_maskEdgeMap[variableIdx + W*H];
#               endif	
#           endif
            //////////////////////////////////////////////////////////////////
            //                   smoothness term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION

            float d  = readX(state, posy, posx, W);
            float d0 = readX(state, posy, posx - 1, W);
            float d1 = readX(state, posy, posx + 1, W);
            float d2 = readX(state, posy - 1, posx, W);
            float d3 = readX(state, posy + 1, posx, W);

            if (IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
                && abs(d - d0)<DEPTH_DISCONTINUITY_THRE
                && abs(d - d1)<DEPTH_DISCONTINUITY_THRE
                && abs(d - d2)<DEPTH_DISCONTINUITY_THRE
                && abs(d - d3)<DEPTH_DISCONTINUITY_THRE)
            {
                E_s = sqMagnitude(4.0f*point(d, posx, posy, input) - (point(d1, posx + 1, posy, input) + point(d0, posx - 1, posy, input) + point(d3, posx, posy + 1, input) + point(d2, posx, posy - 1, input)));
            }
#           endif


#           if USE_DEPTH_CONSTRAINT						
            //position term 			
            E_p = XC - targetDepth;
#           endif



            //////////////////////////////////////////////////////////////////
            //                   piror term
            /////////////////////////////////////////////////////////////////
            //first: calculate the normal for PriorDepth
#           if USE_TEMPORAL_CONSTRAINT
            // TODO: implement
#           endif
            cost = (parameters.weightRegularizer   * E_s*E_s) + // This is usually on the order of 1/10,000,000
                (parameters.weightFitting       * E_p*E_p) + // 
                (parameters.weightShading       * (E_g_h*E_g_h + E_g_v*E_g_v)) +
                (parameters.weightPrior * (E_r_h*E_r_h + E_r_v*E_r_v + E_r_d*E_r_d));
            //cost = readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx);
        }
    }


    return cost;

}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	
    return 0.0f;
    /*
    state.d_delta[variableIdx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 b   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pre = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	// reg/pos
	float4 p = state.d_x[get1DIdx(i, j, input.width, input.height)];
	float4 t = state.d_target[get1DIdx(i, j, input.width, input.height)];
	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0){ float4 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; float4 tq = state.d_target[get1DIdx(n0_i, n0_j, input.width, input.height)]; e_reg += (p - q) - (t - tq); pre += make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN1){ float4 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; float4 tq = state.d_target[get1DIdx(n1_i, n1_j, input.width, input.height)]; e_reg += (p - q) - (t - tq); pre += make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN2){ float4 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; float4 tq = state.d_target[get1DIdx(n2_i, n2_j, input.width, input.height)]; e_reg += (p - q) - (t - tq); pre += make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	if (validN3){ float4 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; float4 tq = state.d_target[get1DIdx(n3_i, n3_j, input.width, input.height)]; e_reg += (p - q) - (t - tq); pre += make_float4(1.0f, 1.0f, 1.0f, 1.0f); }
	b += -e_reg;

	// Preconditioner
	if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
	else					   pre = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	state.d_preconditioner[variableIdx] = pre;
	
	return b;
    */
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    return 0.0f;
    /*
	float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
	const int n0_i = i;		const int n0_j = j - 1; const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height);
	const int n1_i = i;		const int n1_j = j + 1; const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height);
	const int n2_i = i - 1; const int n2_j = j;		const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height);
	const int n3_i = i + 1; const int n3_j = j;		const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height);

	float4 p = state.d_p[get1DIdx(i, j, input.width, input.height)];

	// pos/reg
	float4 e_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (validN0) e_reg += (p - state.d_p[get1DIdx(n0_i, n0_j, input.width, input.height)]);
	if (validN1) e_reg += (p - state.d_p[get1DIdx(n1_i, n1_j, input.width, input.height)]);
	if (validN2) e_reg += (p - state.d_p[get1DIdx(n2_i, n2_j, input.width, input.height)]);
	if (validN3) e_reg += (p - state.d_p[get1DIdx(n3_i, n3_j, input.width, input.height)]);
	b += e_reg;

	return b;
    */
}

#endif

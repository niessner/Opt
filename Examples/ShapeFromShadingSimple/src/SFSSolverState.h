#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 
#include "CameraParams.h"
#include "cuda_SimpleMatrixUtil.h"

struct SolverInput
{
    // Size of optimization domain
    unsigned int N;					// Number of variables

    unsigned int width;				// Image width
    unsigned int height;			// Image height

    // Target
    float* d_targetIntensity;		// Constant target values
    float* d_targetDepth;			// Constant target values

    float* d_depthMapRefinedLastFrameFloat; // refined result of last frame

    // mask edge map
    unsigned char* d_maskEdgeMap;

    // Lighting
    float* d_litcoeff;

    //camera intrinsic parameter
    CameraParams calibparams;

    float4x4 deltaTransform;

};



struct SolverState
{
    // State of the GN Solver
    float*	d_delta;
    float*  d_x;

    float*	d_r;
    float*	d_z;
    float*	d_p;

    float*	d_Ap_X;

    float*	d_scanAlpha;
    float*	d_scanBeta;
    float*	d_rDotzOld;

    float*	d_preconditioner;

    float*	d_sumResidual;
};

#endif

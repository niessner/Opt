#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 

#include "../CameraParams.h"

struct SolverInput
{
	// Size of optimization domain
	unsigned int N;					// Number of variables

	unsigned int width;				// Image width
	unsigned int height;			// Image height
	
	// Target
	float* d_targetIntensity;		// Constant target values
	float* d_targetDepth;			// Constant target values
	float4* d_targetColor;			// Constant target values
	
	float* d_depthMapRefinedLastFrameFloat; // refined result of last frame

	float* d_depthMapMaskFloat;     // mask of foreground

	// Reflectance
	float4* d_targetAlbedo;

	// Lighting
	float* d_litcoeff;
	float* d_litprior;

	int* d_remapArray;

	//camera intrinsic parameter
	CameraParams calibparams;

	bool m_useRemapping;
};

struct SolverState
{
	// State of the GN Solver
	float*	d_delta;					// Current linear update to be computed
	float*  d_x;						// Current state

	float*	d_r;						// Residuum
	float*	d_z;						// Predconditioned residuum
	float*	d_p;						// Decent direction
	
	float*	d_Ap_X;						// Cache values for next kernel call after A = J^T x J x p

	float*	d_scanAlpha;				// Tmp memory for alpha scan
	float*	d_scanBeta;					// Tmp memory for beta scan

	float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)
	
	float*	d_precondioner;				// Preconditioner for linear system

	//add by chenglei
	float* d_grad;
	float* d_shadingdif;
};

#endif

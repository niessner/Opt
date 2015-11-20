#pragma once

#ifndef _PATCH_SOLVER_STATE_
#define _PATCH_SOLVER_STATE_

#include <cuda_runtime.h> 

#include "../CameraParams.h"

#include "../../cuda_SimpleMatrixUtil.h"

#define RGB_RANGE_SCALE 1.0f // leave that at one, doesnt work anymore!
#define DEPTH_RESCALE 1.0f   // leave that at one, doesnt work anymore!

struct PatchSolverInput
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

	// reflectance
	float4* d_albedo;

	int* d_remapArray;

	//camera intrinsic parameter
	CameraParams calibparams;

	float4x4 deltaTransform;

	bool m_useRemapping;
};

struct PatchSolverState
{
	// State of the GN Solver
	float*	d_delta;					// Current linear update to be computed
	float*	d_x;						// Current State
	float*	d_residual;					// Convergence Analysis
};

#endif

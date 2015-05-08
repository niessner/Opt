#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 

#include "CameraParams.h"

struct SolverSHInput
{
	// Size of optimization domain
	unsigned int N;					// Number of variables

	unsigned int width;				// Image width
	unsigned int height;			// Image height
	
	// Target
	float*  d_targetIntensity;		// Constant target values
	float*  d_targetDepth;			// Constant target values
	float4* d_targetColor;			// Constant target values

	// Reflectance
	float4* d_targetAlbedo;

	// Lighting
	float* d_litcoeff;
	float* d_litprior;

	CameraParams calibparams;
};

#endif

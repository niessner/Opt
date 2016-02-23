#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 

#ifndef MINF
#ifdef __CUDACC__
#define MINF __int_as_float(0xff800000)
#else
#define  MINF (std::numeric_limits<float>::infinity())
#endif
#endif 

//#define Stereo_ENABLED
#define LE_THREAD_SIZE 16

struct SolverInput
{
	// Size of optimization domain
	unsigned int N;					// Number of variables

	unsigned int width;				// Image width
	unsigned int height;			// Image height
};



struct SolverState
{
	// State of the GN Solver
	float3*	 d_delta;					
	float3*  d_x;						
	float3*  d_target;					
		 
	float3*	d_r;						
	float3*	d_z;						
	float3*	d_p;						
		 
	float3*	d_Ap_X;						
	
	float*	d_scanAlpha;				
	float*	d_scanBeta;					
	float*	d_rDotzOld;					
	
	float3*	d_precondioner;				

	float*	d_sumResidual;				

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}
};

#endif

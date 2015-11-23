#pragma once

#ifndef SolverSFSState_h
#define SolverSFSState_h

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

// SolverInput is same as PatchSolverInput

struct SolverState
{
	// State of the GN Solver
    float*	 d_delta;
    float*  d_x;
    float*  d_target;
	float*	 d_mask;
		
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

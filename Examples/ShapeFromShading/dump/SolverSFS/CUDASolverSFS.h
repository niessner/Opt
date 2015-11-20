#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <conio.h>

#include "../../cudaUtil.h"
#include "SolverSFSParameters.h"
#include "SolverSFSState.h"
#include "../ICUDASolverSFS.h"
#include "../CUDASolverSHLighting.h"
#include "../CUDAScan.h"

class CUDASolverSFS : public ICUDASolverSFS
{
	public:

		CUDASolverSFS(const Matrix4f& intrinsics, unsigned int imageWidth, unsigned int imageHeight);
		~CUDASolverSFS();

		void solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, float* d_litcoeff, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightFittingIncrement, float weightShading, float weightBoundary, float weightRegularizer, bool useRemapping, float* d_outputDepth);
		void solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, float* d_litcoeff, float4* d_albedos, unsigned char* d_maskEdgeMap,unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightFittingIncrement, float weightShading, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* d_outputDepth);

		void transferDetailSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, Matrix4f rigidM,float* d_depthMapMaskFloat, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* outputDepth);
		
	private:

		int* d_decissionArrayDepthMask;
		int* d_prefixSumDepthMask;
		int* d_remappingArrayDepthMask;

		CUDAScan		m_scan;

		SolverInput m_solverInput;
		SolverState	m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;
};

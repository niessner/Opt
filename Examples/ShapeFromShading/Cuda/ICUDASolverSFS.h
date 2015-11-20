#pragma once

#include <vector>
#include "../../Eigen.h"

class ICUDASolverSFS
{
	public:

		ICUDASolverSFS(const Matrix4f& intrinsics) : m_intrinsics(intrinsics)
		{
		}

		virtual ~ICUDASolverSFS()
		{
		}
		
		virtual void solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, mat4f& deltaTransform, float* d_litcoeff, float4* d_albedos, unsigned char* d_maskEdgeMap,unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFittingStart, float weightFittingIncrement, float weightShading, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* d_outputDepth) = 0;

	protected:

		Matrix4f m_intrinsics;
};

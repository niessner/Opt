#pragma once
/** TODO: Refactor and rename this to reflect the fact that all optimization variants run through this code path */
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "../../cudaUtil.h"
#include "PatchSolverSFSParameters.h"
#include "PatchSolverSFSState.h"
#include "../SolverSFS/SolverSFSState.h"
#include "../ICUDASolverSFS.h"
#include "../CUDAScan.h"
#include "../../ConvergenceAnalysis.h"
struct Plan;
class Optimizer;
class CUDAPatchSolverSFS : public ICUDASolverSFS
{
	public:
        static Optimizer* s_optimizerNoAD;
        static Optimizer* s_optimizerAD;

		CUDAPatchSolverSFS(const Matrix4f& intrinsics, unsigned int imageWidth, unsigned int imageHeight, unsigned int level);
		~CUDAPatchSolverSFS();
        
		void solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, mat4f& deltaTransform, float *d_litcoeff, float4 *d_albedos, unsigned char* d_maskEdgeMap, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightShadingIncrement, float weightShadingStart, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* outputDepth);

        Plan* maybeInitOptimizerAndPlan(bool useAD, const std::string& terraFile, const std::string& solverName, int width, int height, const std::vector<uint32_t>& elemsize);
	
        void resetPlan();
    private:

		int* d_decissionArrayDepthMask;
		int* d_prefixSumDepthMask;
		int* d_remappingArrayDepthMask;

		CUDAScan		m_scan;

		PatchSolverInput m_solverInput;
		PatchSolverState m_patchSolverState;
        SolverState      m_solverState;

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;

		unsigned int m_level;

        Plan* m_planNoAD;
        Plan* m_planAD;
        
};

#include "stdafx.h"
#include "../CUDASolverSHLighting.h"

#include "TerraSolverParameters.h"
#include "CUDAPatchSolverSFS.h"
#include "../CUDASolverSHLighting.h"
#include "../../GlobalAppState.h"
#include "../../Optimizer.h"
#include "../../DumpOptImage.h"

extern "C" void copyFloatMapFill(float* d_output, float* d_input, unsigned int width, unsigned int height);

extern "C" void patchSolveSFSStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, ConvergenceAnalysis<float>* ca);
extern "C" void solveSFSStub(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, ConvergenceAnalysis<float>* ca);

extern "C" void patchSolveSFSEvalCurrentCostJTFPreAndJTJStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult);
//extern "C" void solveSFSEvalCurrentCostJTFPreAndJTJStub(PatchSolverInput& input, SolverState& state, PatchSolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult);

extern "C" void clearDecissionArrayPatchDepthMask(int* d_output, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void computeDecissionArrayPatchDepthMask(int* d_output, float* d_input, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void computeRemappingArrayPatchDepthMask(int* d_output, float* d_input, int* d_prefixSum, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void DebugPatchRemapArray(float* d_mask, int* d_remapArray, unsigned int patchSize, unsigned int numElements, unsigned int inputWidth, unsigned int inputHeight);

Optimizer* CUDAPatchSolverSFS::s_optimizerNoAD  = NULL;
Optimizer* CUDAPatchSolverSFS::s_optimizerAD    = NULL;

CUDAPatchSolverSFS::CUDAPatchSolverSFS(const Matrix4f& intrinsics, unsigned int imageWidth, unsigned int imageHeight, unsigned int level) : m_imageWidth(imageWidth), m_imageHeight(imageHeight), ICUDASolverSFS(intrinsics), m_planAD(NULL), m_planNoAD(NULL)
{
	const unsigned int numberOfVariables = m_imageWidth*m_imageHeight;

	cutilSafeCall(cudaMalloc(&d_decissionArrayDepthMask, sizeof(int)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&d_prefixSumDepthMask, sizeof(int)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&d_remappingArrayDepthMask, sizeof(int)*numberOfVariables));

	m_solverInput.N = m_imageWidth*m_imageHeight;
	m_solverInput.width = m_imageWidth;
	m_solverInput.height = m_imageHeight;
	m_solverInput.d_remapArray = NULL;

	m_solverInput.calibparams.fx =  m_intrinsics(0, 0);
	m_solverInput.calibparams.fy = -m_intrinsics(1, 1);
	m_solverInput.calibparams.ux =  m_intrinsics(0, 3);
	m_solverInput.calibparams.uy =  m_intrinsics(1, 3);

	// State
	cutilSafeCall(cudaMalloc(&m_patchSolverState.d_delta, sizeof(float)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_patchSolverState.d_residual, sizeof(float)*2));

    const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
    const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

    // Non-patch State
    cutilSafeCall(cudaMalloc(&m_solverState.d_delta, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_r, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_z, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_p, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_X, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float)*tmpBufferSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta, sizeof(float)*tmpBufferSize));
    cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_preconditioner, sizeof(float)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));

	m_level = level;
}

CUDAPatchSolverSFS::~CUDAPatchSolverSFS()
{
	cutilSafeCall(cudaFree(d_decissionArrayDepthMask));
	cutilSafeCall(cudaFree(d_prefixSumDepthMask));
	cutilSafeCall(cudaFree(d_remappingArrayDepthMask));

	// State
	cutilSafeCall(cudaFree(m_patchSolverState.d_delta));
	cutilSafeCall(cudaFree(m_patchSolverState.d_residual));


    // Non-patch State
    cutilSafeCall(cudaFree(m_solverState.d_delta));
    cutilSafeCall(cudaFree(m_solverState.d_r));
    cutilSafeCall(cudaFree(m_solverState.d_z));
    cutilSafeCall(cudaFree(m_solverState.d_p));
    cutilSafeCall(cudaFree(m_solverState.d_Ap_X));
    cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
    cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
    cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
    cutilSafeCall(cudaFree(m_solverState.d_preconditioner));
    cutilSafeCall(cudaFree(m_solverState.d_sumResidual));


}


enum SolveMode {TERRA_NO_AD, TERRA_AD, CUDA};


static void constructTerraInput(void* d_x, const PatchSolverInput& solverInput, const PatchSolverParameters& parameters, float* deltaTransformPtr, std::vector<void*>& images, TerraSolverParameters& tParams, bool optCPU) {
    
    if (optCPU) {
        int numBytes = sizeof(float)*solverInput.width * solverInput.height;
        static void* x = malloc(numBytes);
        static void* targetDepth = malloc(numBytes);
        static void* targetIntensity = malloc(numBytes);
        static void* depthMapRefinedLastFrameFloat = malloc(numBytes);
        static void* maskEdgeMap = malloc(numBytes);
        cudaMemcpy(x, d_x, numBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(targetDepth, solverInput.d_targetDepth, numBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(targetIntensity, solverInput.d_targetIntensity, numBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(depthMapRefinedLastFrameFloat, solverInput.d_depthMapRefinedLastFrameFloat, numBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(maskEdgeMap, solverInput.d_maskEdgeMap, 2 * solverInput.width * solverInput.height, cudaMemcpyDeviceToHost);
        images.push_back(x);
        images.push_back(targetDepth);
        images.push_back(targetIntensity);
        images.push_back(depthMapRefinedLastFrameFloat);
        images.push_back(maskEdgeMap);
        images.push_back((char*)(maskEdgeMap) + (solverInput.width * solverInput.height));
    } else {
        images.push_back(d_x);
        images.push_back(solverInput.d_targetDepth);
        images.push_back(solverInput.d_targetIntensity);
        //image.push_back(solverInput.d_remapArray);
        images.push_back(solverInput.d_depthMapRefinedLastFrameFloat);
        images.push_back(solverInput.d_maskEdgeMap); // row
        images.push_back(solverInput.d_maskEdgeMap + (solverInput.width * solverInput.height)); // col
    }

    tParams = TerraSolverParameters(parameters, solverInput.calibparams, deltaTransformPtr, solverInput.d_litcoeff);
}

Plan* CUDAPatchSolverSFS::maybeInitOptimizerAndPlan(bool useAD, const std::string& terraFile, const std::string& solverName, int width, int height, const std::vector<uint32_t>& elemsize) {
    if (useAD) {
        if (s_optimizerAD == NULL) {
            s_optimizerAD = new Optimizer();
            s_optimizerAD->defineProblem(terraFile, solverName);
        }

        if (m_planAD == NULL) {
            m_planAD = s_optimizerAD->planProblem(width, height, elemsize);
        }
        return m_planAD;
    }
    if (s_optimizerNoAD == NULL) {
        s_optimizerNoAD = new Optimizer();
        s_optimizerNoAD->defineProblem(terraFile, solverName);
    }

    if (m_planNoAD == NULL) {
        m_planNoAD = s_optimizerNoAD->planProblem(width, height, elemsize);
    }
    return m_planNoAD;
    
}

void CUDAPatchSolverSFS::resetPlan() {
    if (m_planAD != NULL) {
        Opt_PlanFree(s_optimizerAD->state(), m_planAD);
        m_planAD = NULL;
    }
    if (m_planNoAD != NULL) {
        Opt_PlanFree(s_optimizerAD->state(), m_planNoAD);
        m_planNoAD = NULL;
    }
}

#define OPT_CPU 0

void CUDAPatchSolverSFS::solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, mat4f& deltaTransform, float *d_litcoeff, float4 *d_albedos, unsigned char* d_maskEdgeMap, unsigned int nNonLinearIterations, unsigned int nLinearIterations, unsigned int nPatchIterations, float weightFitting, float weightShadingIncrement, float weightShadingStart, float weightBoundary, float weightRegularizer, float weightPrior, bool useRemapping, float* outputDepth)
{
    unsigned int numElements = 0;
    if (useRemapping)
    {
        clearDecissionArrayPatchDepthMask(d_decissionArrayDepthMask, m_solverInput.width, m_solverInput.height);
        computeDecissionArrayPatchDepthMask(d_decissionArrayDepthMask, d_depthMapMaskFloat, 16, m_solverInput.width, m_solverInput.height); // keep consistent with GPU
        numElements = m_scan.prefixSum(m_solverInput.width*m_solverInput.height, d_decissionArrayDepthMask, d_prefixSumDepthMask);
        computeRemappingArrayPatchDepthMask(d_remappingArrayDepthMask, d_depthMapMaskFloat, d_prefixSumDepthMask, 16, m_solverInput.width, m_solverInput.height);
    }

    m_patchSolverState.d_x  = outputDepth;
    m_solverState.d_x       = outputDepth;

    m_solverInput.d_targetIntensity = d_targetIntensity;
    m_solverInput.d_targetDepth = d_targetDepth;
    m_solverInput.d_depthMapRefinedLastFrameFloat = d_depthMapRefinedLastFrameFloat;
    m_solverInput.d_maskEdgeMap = d_maskEdgeMap;
    m_solverInput.d_litcoeff = d_litcoeff;
    m_solverInput.d_albedo = d_albedos; // NULL
    m_solverInput.d_remapArray = d_remappingArrayDepthMask;
    m_solverInput.deltaTransform = float4x4(deltaTransform.ptr()); // transformation to last frame

    if (useRemapping) m_solverInput.N = numElements;
    m_solverInput.m_useRemapping = useRemapping;

    PatchSolverParameters parameters;
    parameters.weightFitting = weightFitting;
    parameters.weightShadingStart = weightShadingStart;
    parameters.weightShadingIncrement = weightShadingIncrement;
    parameters.weightShading = weightShadingStart + weightShadingIncrement * nLinearIterations; // default to final value
    parameters.weightRegularizer = weightRegularizer;
    parameters.weightBoundary = weightBoundary;
    parameters.weightPrior = weightPrior;
    parameters.nNonLinearIterations = nNonLinearIterations;
    parameters.nLinIterations = nLinearIterations;
    parameters.nPatchIterations = nPatchIterations;

    ConvergenceAnalysis<float>* ca = GlobalAppState::get().s_convergenceAnalysisIsRunning && GlobalAppState::get().s_playData ? &GlobalAppState::get().s_convergenceAnalysis : NULL;
    SolveMode mode = (GlobalAppState::get().s_optimizer == 0) ? CUDA : ((GlobalAppState::get().s_optimizer == 1) ? SolveMode::TERRA_AD : SolveMode::TERRA_NO_AD);
    std::vector<uint32_t> elemsize;
    for (int i = 0; i < 4; ++i) {
        elemsize.push_back(sizeof(float));
    }
    for (int i = 0; i < 2; ++i) {
        elemsize.push_back(sizeof(char));
    }
    bool saveJTFAndPreAndJTJ = false;
    switch (mode) {
    case CUDA:
        {
            
            if (saveJTFAndPreAndJTJ) {
                float *jtfResult;
                float *preResult;
                float *jtjResult;
                float *costResult;
                int numberOfVariables = m_solverInput.width*m_solverInput.height;
                cutilSafeCall(cudaMalloc(&jtfResult, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMalloc(&preResult, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMalloc(&jtjResult, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMalloc(&costResult, sizeof(float)*numberOfVariables));

                cutilSafeCall(cudaMemset(jtfResult, 0, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMemset(preResult, 0, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMemset(jtjResult, 0, sizeof(float)*numberOfVariables));
                cutilSafeCall(cudaMemset(costResult, 0, sizeof(float)*numberOfVariables));

                patchSolveSFSEvalCurrentCostJTFPreAndJTJStub(m_solverInput, m_patchSolverState, parameters, costResult, jtfResult, preResult, jtjResult);

                OptUtil::dumpOptImage(costResult, "cost_cuda.imagedump", m_solverInput.width, m_solverInput.height, 1);
                OptUtil::dumpOptImage(jtfResult, "JTF_cuda.imagedump", m_solverInput.width, m_solverInput.height, 1);
                OptUtil::dumpOptImage(preResult, "Pre_cuda.imagedump", m_solverInput.width, m_solverInput.height, 1);
                OptUtil::dumpOptImage(jtjResult, "JTJ_cuda.imagedump", m_solverInput.width, m_solverInput.height, 1);
                cutilSafeCall(cudaFree(costResult));
                cutilSafeCall(cudaFree(jtfResult));
                cutilSafeCall(cudaFree(preResult));
                cutilSafeCall(cudaFree(jtjResult));
            }

            if (GlobalAppState::get().s_useBlockSolver) {
                patchSolveSFSStub(m_solverInput, m_patchSolverState, parameters, ca);
            } else {
                solveSFSStub(m_solverInput, m_solverState, parameters, ca);
            }
        }
        break;
    case TERRA_AD: 
        {
#           if OPT_CPU
                Plan* plan = maybeInitOptimizerAndPlan(true, "shapeFromShadingAD.t", "gradientDescentCPU", m_solverInput.width, m_solverInput.height, elemsize);
#           else
            Plan* plan = maybeInitOptimizerAndPlan(true, "shapeFromShadingAD.t", GlobalAppState::get().s_useBlockSolver ? "gaussNewtonBlockGPU" : "gaussNewtonGPU", m_solverInput.width, m_solverInput.height, elemsize);
#           endif

            std::vector<void*> images;
            TerraSolverParameters tParams;
            
            constructTerraInput(m_patchSolverState.d_x, m_solverInput, parameters, deltaTransform.ptr(), images, tParams, OPT_CPU == 1 ? true : false);
            TerraSolverParameterPointers indirectParameters(tParams);
            s_optimizerAD->solve(plan, images, &indirectParameters);
        }
        break;
    case TERRA_NO_AD:
        {
#           if OPT_CPU
                Plan* plan = maybeInitOptimizerAndPlan(false, "shapeFromShading.t", "gradientDescentCPU", m_solverInput.width, m_solverInput.height, elemsize);
#           else
            Plan* plan = maybeInitOptimizerAndPlan(false, "shapeFromShading.t", GlobalAppState::get().s_useBlockSolver ? "gaussNewtonBlockGPU" : "gaussNewtonGPU", m_solverInput.width, m_solverInput.height, elemsize);
#           endif
            std::vector<void*> images;
            TerraSolverParameters tParams;
            constructTerraInput(m_patchSolverState.d_x, m_solverInput, parameters, deltaTransform.ptr(), images, tParams, OPT_CPU == 1 ? true : false);
            TerraSolverParameterPointers indirectParameters(tParams);
            s_optimizerNoAD->solve(plan, images, &indirectParameters);
        }
        break;
    default:
        fprintf(stderr, "You suck\n");
    }
    
}

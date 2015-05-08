#ifndef SFSHelpers_h
#define SFSHelpers_h

#include <cuda_runtime.h>
#include "cutil_inline.h"
#include "SolverSHLightingState.h"
#include "CUDASolverSHLighting.h"
class SFSHelpers {

    shared_ptr<CUDASolverSHLighting> m_shLightingSolver;

public:

    SFSHelpers() : m_shLightingSolver(NULL) {}

    void solveLighting(float* d_targetDepth, float* d_targetIntensity, float4* d_normalMap, float* d_litprior, float* d_litcoeff, float thres_depth, int width, int height)
    {
        if (!m_shLightingSolver) {
            m_shLightingSolver = shared_ptr<CUDASolverSHLighting>(new CUDASolverSHLighting(width, height));
        }
        SolverSHInput solverInput;
        solverInput.N = width * height;
        solverInput.width = width;
        solverInput.height = height;
        solverInput.d_targetIntensity = d_targetIntensity;
        solverInput.d_targetDepth = d_targetDepth;
        solverInput.d_litcoeff = d_litcoeff;
        solverInput.d_litprior = d_litprior;

        solverInput.calibparams.fx = 532.569458;//m_intrinsics(0, 0);
        solverInput.calibparams.fy = 531.541077;//-m_intrinsics(1, 1);
        solverInput.calibparams.ux = 320.0;//m_intrinsics(0, 3);
        solverInput.calibparams.uy = 240.0;//m_intrinsics(1, 3);

        m_shLightingSolver->solveLighting(solverInput, d_normalMap, thres_depth);
    }



    void solveReflectance(float* d_targetDepth, float4* d_targetColor, float4* d_normalMap, float* d_litcoeff, float4* d_albedos, int width, int height)
    {
        cutilSafeCall(cudaMemset(d_albedos, 0, width * height * sizeof(float4)));

        SolverSHInput solverInput;
        solverInput.N = width * height;
        solverInput.width = width;
        solverInput.height = height;
        solverInput.d_targetColor = d_targetColor;
        solverInput.d_targetDepth = d_targetDepth;
        solverInput.d_targetAlbedo = d_albedos;
        solverInput.d_litcoeff = d_litcoeff;

        m_shLightingSolver->solveReflectance(solverInput, d_normalMap);
    }

    void initializeLighting(float* d_litcoeff)
    {
        float h_litestmat[9];
        h_litestmat[0] = 1.0f; h_litestmat[1] = 0.0f; h_litestmat[2] = -0.5f; h_litestmat[3] = 0.0f; h_litestmat[4] = 0.0f; h_litestmat[5] = 0.0f; h_litestmat[6] = 0.0f; h_litestmat[7] = 0.0f; h_litestmat[8] = 0.0f;

        cutilSafeCall(cudaMemcpy(d_litcoeff, h_litestmat, 9 * sizeof(float), cudaMemcpyHostToDevice));
    }

};

#endif
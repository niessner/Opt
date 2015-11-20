#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "SolverSHLightingState.h"
#include "../../cudaUtil.h"
#include "PatchSolverSFS/CUDAPatchSolverSFS.h"
#include "../DX11QuadDrawer.h"
#include "../GlobalAppState.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "CUDASolverSHLighting.h"



extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, float* d_depthMaskMap);

extern "C" void downsampleUnsignedCharMap(unsigned char* d_MapResampled, unsigned int outputWidth, unsigned int outputHeight, unsigned char* d_Map, unsigned int inputWidth, unsigned int inputHeight);

class CUDAHierarchicalSolverSFS
{
	public:

		CUDAHierarchicalSolverSFS(const Matrix4f& intrinsics, std::vector<unsigned int>& optimizationWidthAtLevel, std::vector<unsigned int>& optimizationHeightAtLevel, unsigned int numLevels) : m_obj_lit(optimizationWidthAtLevel[0], optimizationHeightAtLevel[0]), m_intrinsics(intrinsics)
		{
			allocateHierarchy(optimizationWidthAtLevel, optimizationHeightAtLevel, numLevels);
		}

		~CUDAHierarchicalSolverSFS()
		{
			deallocateHierarchy();
		}

        void resetOptPlans() {
            for (unsigned int i = 0; i < m_solversSFS.size(); i++)
            {
                ((CUDAPatchSolverSFS*)m_solversSFS[i])->resetPlan();
            }
        }

		void allocateHierarchy(std::vector<unsigned int>& optimizationWidthAtLevel, std::vector<unsigned int>& optimizationHeightAtLevel, unsigned int numLevels)
		{
			unsigned int originalWidth = optimizationWidthAtLevel[0];
			unsigned int originalHeight = optimizationHeightAtLevel[0];

			m_widthAtLevel.resize(numLevels);
			m_heightAtLevel.resize(numLevels);

			m_solversSFS.resize(numLevels);
			
			m_targetIntensityAtLevel.resize(numLevels);
			m_targetDepthAtLevel.resize(numLevels);
			m_targetDepthMaskAtLevel.resize(numLevels);
			m_depthMapRefinedLastFrameAtLevel.resize(numLevels);
			m_outputDepthAtLevel.resize(numLevels);

			m_maskEdgeMapAtLevel.resize(numLevels);
			
			for(unsigned int i = 0; i < numLevels; i++)
			{
				m_widthAtLevel[i] = optimizationWidthAtLevel[i];
				m_heightAtLevel[i] = optimizationHeightAtLevel[i];

				const float scaleWidth = (float)optimizationWidthAtLevel[i]/(float)originalWidth;
				const float scaleHeight = (float)optimizationHeightAtLevel[i]/(float)originalHeight;

				Matrix4f newIntrinsics = m_intrinsics;
				newIntrinsics(0, 0) *= scaleWidth; newIntrinsics(1, 1) *= scaleHeight; newIntrinsics(0, 3) *= scaleWidth; newIntrinsics(1, 3) *= scaleHeight;

				m_solversSFS[i] = new CUDAPatchSolverSFS(newIntrinsics, m_widthAtLevel[i], m_heightAtLevel[i], i);

				if(i != 0)
				{
					cutilSafeCall(cudaMalloc(&(m_targetIntensityAtLevel[i]),		  m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float)));
					cutilSafeCall(cudaMalloc(&(m_targetDepthAtLevel[i]),			  m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float)));
					cutilSafeCall(cudaMalloc(&(m_targetDepthMaskAtLevel[i]),		  m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float)));
					cutilSafeCall(cudaMalloc(&(m_depthMapRefinedLastFrameAtLevel[i]), m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float)));
					cutilSafeCall(cudaMalloc(&(m_outputDepthAtLevel[i]),			  m_widthAtLevel[i]*m_heightAtLevel[i]*sizeof(float)));

					cutilSafeCall(cudaMalloc(&(m_maskEdgeMapAtLevel[i]),			  m_widthAtLevel[i]*m_heightAtLevel[i]*2*sizeof(unsigned char)));
				}
			}
		}

		void deallocateHierarchy()
		{
			for(unsigned int i = 0; i < m_solversSFS.size(); i++)
			{
				SAFE_DELETE(m_solversSFS[i]);
			}

			for(unsigned int i = 0; i < m_targetIntensityAtLevel.size(); i++)
			{
				if(i != 0) cutilSafeCall(cudaFree(m_targetIntensityAtLevel[i]));
				if(i != 0) cutilSafeCall(cudaFree(m_targetDepthAtLevel[i]));
				if(i != 0) cutilSafeCall(cudaFree(m_targetDepthMaskAtLevel[i]));
				if(i != 0) cutilSafeCall(cudaFree(m_depthMapRefinedLastFrameAtLevel[i]));
				if(i != 0) cutilSafeCall(cudaFree(m_outputDepthAtLevel[i]));

				if(i != 0) cutilSafeCall(cudaFree(m_maskEdgeMapAtLevel[i]));
			}
		}


		void solveLighting(float* d_targetDepth, float* d_targetIntensity, float4* d_normalMap, float* d_litprior, float* d_litcoeff, float thres_depth)
		{
			SolverSHInput solverInput;
			solverInput.N = m_widthAtLevel[0]*m_heightAtLevel[0];
			solverInput.width =  m_widthAtLevel[0];
			solverInput.height = m_heightAtLevel[0];
			solverInput.d_targetIntensity = d_targetIntensity;
			solverInput.d_targetDepth = d_targetDepth;
			solverInput.d_litcoeff = d_litcoeff;
			solverInput.d_litprior = d_litprior;			
			
			solverInput.calibparams.fx =  m_intrinsics(0, 0);
			solverInput.calibparams.fy =  -m_intrinsics(1, 1);
			solverInput.calibparams.ux =  m_intrinsics(0, 3);
			solverInput.calibparams.uy =  m_intrinsics(1, 3);
						
			m_obj_lit.solveLighting(solverInput, d_normalMap, thres_depth);
		}


		
		void solveReflectance(float* d_targetDepth, float4* d_targetColor, float4* d_normalMap, float* d_litcoeff, float4* d_albedos)
		{
			cutilSafeCall(cudaMemset(d_albedos, 0, m_widthAtLevel[0]*m_heightAtLevel[0]*sizeof(float4)));

			SolverSHInput solverInput;
			solverInput.N = m_widthAtLevel[0]*m_heightAtLevel[0];
			solverInput.width =  m_widthAtLevel[0];
			solverInput.height = m_heightAtLevel[0];
			solverInput.d_targetColor = d_targetColor;
			solverInput.d_targetDepth = d_targetDepth;
			solverInput.d_targetAlbedo = d_albedos;
			solverInput.d_litcoeff = d_litcoeff;			
		
			m_obj_lit.solveReflectance(solverInput, d_normalMap);
		}

		void initializeLighting(float* d_litcoeff)
		{
			float h_litestmat[9];
			h_litestmat[0] = 1.0f; h_litestmat[1] = 0.0f; h_litestmat[2] = -0.5f;h_litestmat[3] = 0.0f; h_litestmat[4] = 0.0f;h_litestmat[5] = 0.0f;h_litestmat[6] =  0.0f;h_litestmat[7] = 0.0f;h_litestmat[8] = 0.0f;
			
			cutilSafeCall(cudaMemcpy(d_litcoeff,h_litestmat,9*sizeof(float),cudaMemcpyHostToDevice));
		}
		
		void solveSFS(float* d_targetDepth, float* d_depthMapRefinedLastFrameFloat, float* d_depthMapMaskFloat, float* d_targetIntensity, unsigned char* d_maskEdgeMap, mat4f& deltaTransform, std::vector<unsigned int>& nNonLinearIterations, std::vector<unsigned int>& nLinearIterations, std::vector<unsigned int>& nPatchIterations, std::vector<float>& weightFitting, std::vector<float>& weightShadingIncrement, std::vector<float>& weightShadingStart, std::vector<float>& weightBoundary, std::vector<float>& weightRegularizer, std::vector<float>& weightPrior, float* d_litcoeff, float4* d_albedos, float* d_outputDepth, bool refineOnlyForeground)
		{			
			m_targetIntensityAtLevel[0]			 = d_targetIntensity;
			m_targetDepthAtLevel[0]				 = d_targetDepth;
			m_targetDepthMaskAtLevel[0]			 = d_depthMapMaskFloat;
			m_depthMapRefinedLastFrameAtLevel[0] = d_depthMapRefinedLastFrameFloat;
			m_outputDepthAtLevel[0]				 = d_outputDepth;

			m_maskEdgeMapAtLevel[0]				 = d_maskEdgeMap;

			// Compute restriction of input data
			unsigned int numLevels = (unsigned int)m_targetIntensityAtLevel.size();
			for(unsigned int i = 0; i < numLevels-1; i++)
			{
				resampleFloatMap(m_targetIntensityAtLevel[i+1],	m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_targetIntensityAtLevel[i],	m_widthAtLevel[i], m_heightAtLevel[i], NULL);
				resampleFloatMap(m_targetDepthAtLevel[i+1],		m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_targetDepthAtLevel[i],		m_widthAtLevel[i], m_heightAtLevel[i], NULL);
				
				resampleFloatMap(m_targetDepthMaskAtLevel[i+1], m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_targetDepthMaskAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i], NULL);
				
				resampleFloatMap(m_depthMapRefinedLastFrameAtLevel[i+1], m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_depthMapRefinedLastFrameAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i], NULL);
				resampleFloatMap(m_outputDepthAtLevel[i+1],				 m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_outputDepthAtLevel[i],			  m_widthAtLevel[i], m_heightAtLevel[i], NULL);

				downsampleUnsignedCharMap(m_maskEdgeMapAtLevel[i+1], m_widthAtLevel[i+1], m_heightAtLevel[i+1], m_maskEdgeMapAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i]);
			}

			// Solve coarse to fine
			for(int i = ((int)numLevels)-1; i >= 0; i--)
			{
				m_solversSFS[i]->solveSFS(m_targetDepthAtLevel[i], m_depthMapRefinedLastFrameAtLevel[i], m_targetDepthMaskAtLevel[i], m_targetIntensityAtLevel[i], deltaTransform, d_litcoeff, NULL /*d_albedo*/, m_maskEdgeMapAtLevel[i], nNonLinearIterations[i], nLinearIterations[i], nPatchIterations[i], weightFitting[i], weightShadingIncrement[i], weightShadingStart[i], weightBoundary[i], weightRegularizer[i], weightPrior[i], refineOnlyForeground, m_outputDepthAtLevel[i]);
			
				if(i != 0) resampleFloatMap(m_outputDepthAtLevel[i-1], m_widthAtLevel[i-1], m_heightAtLevel[i-1], m_outputDepthAtLevel[i], m_widthAtLevel[i], m_heightAtLevel[i], m_targetDepthMaskAtLevel[i-1]);
			}

			resampleFloatMap(m_outputDepthAtLevel[0], m_widthAtLevel[0], m_heightAtLevel[0], m_outputDepthAtLevel[0], m_widthAtLevel[0], m_heightAtLevel[0], m_targetDepthMaskAtLevel[0]);

			// Clear the back buffer
			//static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
			//ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
			//ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
			//DXUTGetD3D11DeviceContext()->ClearRenderTargetView(pRTV, ClearColor);
			//DXUTGetD3D11DeviceContext()->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
			//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), DXUTGetD3D11DeviceContext(), m_targetDepthMaskAtLevel[2], 1, m_widthAtLevel[2], m_heightAtLevel[2], 0.2f);
		}

	private:

		CUDASolverSHLighting m_obj_lit;

		std::vector<ICUDASolverSFS*> m_solversSFS;

		std::vector<unsigned int> m_widthAtLevel;
		std::vector<unsigned int> m_heightAtLevel;

		std::vector<float*> m_targetIntensityAtLevel;
		std::vector<float*> m_targetDepthAtLevel;
		std::vector<float*> m_targetDepthMaskAtLevel;
		std::vector<float*> m_depthMapRefinedLastFrameAtLevel;
		std::vector<float*> m_outputDepthAtLevel;

		std::vector<unsigned char*> m_maskEdgeMapAtLevel;

		Matrix4f m_intrinsics;

		int m_FrameIdx;
};

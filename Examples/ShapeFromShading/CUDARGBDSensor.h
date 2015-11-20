#pragma once

#include "RGBDSensor.h"
#include <cuda_runtime.h> 
#include <cstdlib>
#include "cudaUtil.h"
#include <cuda_d3d11_interop.h> 
#include "cuda_SimpleMatrixUtil.h"
#include "Cuda/PatchSolverSFS/CUDAPatchSolverSFS.h"
#include "Cuda/CUDAHierarchicalSolverSFS.h"
#include "GlobalAppState.h"
#include "Timer.h"
#include "CUDAHoleFiller.h"
#include "Cuda\CUDAScan.h"
#include "CUDARGBDAdapter.h"
#include "CUDA/CUDACameraTrackingMultiRes.h"
#include "DX11RGBDRenderer.h"
#include "DX11CustomRenderTarget.h"


extern "C" void computeDerivativesFloat(float* d_outputDU, float* d_outputDV, float* d_input, unsigned int width, unsigned int height);

class CUDARGBDSensor
{
	public:

        void resetOptPlans() {
            m_hierarchicalSolverSFS->resetOptPlans();
        }

		CUDARGBDSensor();
		~CUDARGBDSensor();

		void OnD3D11DestroyDevice();
			
		HRESULT OnD3D11CreateDevice(ID3D11Device* device, CUDARGBDAdapter* CUDARGBDAdapter);

		HRESULT process(ID3D11DeviceContext* context);
		
		//! enables bilateral filtering of the depth value
		void setFiterDepthValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);
		void setFiterIntensityValues(bool b = true, float sigmaD = 1.0f, float sigmaR = 1.0f);

		float* GetFilteredDepthMapMaskFloat();
		float* GetDepthMapRefinedFloat();
		float* GetDepthMapRefinedLastFrameFloat();
		
		float* GetDepthMapRawFloat()
		{
			return m_RGBDAdapter->GetDepthMapResampledFloat();
		}

		float* GetDepthMapColorSpaceFloat();
		float4* GetAlbedoFloat4();
		float* GetShadingFloat();

		float4* GetCameraSpacePositionsFloat4();

		float4* GetColorMapFilteredFloat4();
		float4* GetNormalMapFloat4();
		float4* getNormalMapNoRefinementFloat4() {
			return d_normalMapNoRefinementFloat4;
		}
		float*  GetEdgeMask();

		float* GetIntensityMapFloat();
		float* GetIntensityMapFilteredFloat();
		float* GetHSVMapFloat();

		float4* GetClusterColors();

		float* GetIntensityDerivativeDUMapFloat();
		float* GetIntensityDerivativeDVMapFloat();

		unsigned int getDepthWidth() const;
		unsigned int getDepthHeight() const;
		unsigned int getColorWidth() const;
		unsigned int getColorHeight() const;
		
		bool checkValidT1(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4);
		bool checkValidT2(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4);

		void saveCameraSpaceMapAsMesh(bool onlySaveForeground);
		void saveCameraSpaceMapAsMesh(float4* d_positionData, const std::string& filename, bool onlySaveForeground);

		void toggleFillHoles()
		{
			m_bFillHoles = !m_bFillHoles;
			if(m_bFillHoles)
			{
				std::cout << "hole filling enabled" << std::endl;
			}
			else
			{
				std::cout << "hole filling disabled" << std::endl;
			}
		}

		void toggleSDSRefinement()
		{
			m_bSFSRefinement = !m_bSFSRefinement;
			if (m_bSFSRefinement)
			{
				std::cout << "sfs refinement enabled" << std::endl;
			}
			else
			{
				std::cout << "sfs refinement disabled" << std::endl;
			}
		}

		mat4f trackRigidTransform(float4* d_positionsCurrent, float4* d_normalsCurrent, float4* d_positionsLast, float4* d_normalsLast, const mat4f& intrinsics);

		//! resets the accumulated rigid transform and frees all memory from recordings
		void reset() {
			m_rigidTransformIntegrated.setIdentity();

			for (auto iter = m_RecordedDepthRefined.begin(); iter != m_RecordedDepthRefined.end(); iter++)
				SAFE_DELETE_ARRAY(*iter);
			for (auto iter = m_RecordedDepth.begin(); iter != m_RecordedDepth.end(); iter++) {
				SAFE_DELETE_ARRAY(*iter);
			}
			m_RecordedDepthRefined.clear();
			m_RecordedDepth.clear();

			m_RGBDAdapter->getRGBDSensor()->reset();
		}

		//void saveRecordedFramesToFile(const std::string& filename) {
		void saveRecordedFramesToFile() {
			
			saveRecordedFramesToFile("recordingOriginal.sensor", m_RecordedDepth);
			saveRecordedFramesToFile("recordingRefined.sensor", m_RecordedDepthRefined);

			m_RGBDAdapter->getRGBDSensor()->saveRecordedFramesToFile("recordingRaw.sensor");
		}

		//! chenglei's exr format
		void saveDepthAndColor() {

			std::string outexrpath = "dump\\depth";
			std::string outpngpath = "dump\\color";

			float* d_targetDepth = d_depthMapColorSpaceFloat; 
			float4* d_targetColor = d_colorMapFilteredFloat4;

			unsigned int numpixels = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();

			float *h_depth = new float[numpixels];
			float4 *h_color = new float4[numpixels];
			cutilSafeCall(cudaMemcpy(h_depth, d_targetDepth, sizeof(float)*numpixels,cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_color, d_targetColor, sizeof(float4)*numpixels,cudaMemcpyDeviceToHost));

            /*
			cv::Mat rgbmat(m_RGBDAdapter->getHeight(), m_RGBDAdapter->getWidth(), CV_8UC3);

			for (unsigned int i = 0; i < m_RGBDAdapter->getHeight(); i++) {
				for (unsigned int j=0; j < m_RGBDAdapter->getWidth(); j++) {
					unsigned int tmpind = i*m_RGBDAdapter->getWidth() + j;
					if (h_depth[tmpind] < 0) h_depth[tmpind] = 0.0f;

					cv::Vec3b& rgb = rgbmat.at<cv::Vec3b>(i, j);
					rgb[2] = cvRound(h_color[tmpind].x*255.0f);
					rgb[1] = cvRound(h_color[tmpind].y*255.0f);
					rgb[0] = cvRound(h_color[tmpind].z*255.0f);		
				}
			}
			int sizes[2];
			sizes[0] = m_RGBDAdapter->getHeight();
			sizes[1] = m_RGBDAdapter->getWidth();	
			cv::Mat imageHDR(2, sizes, CV_32FC1, (void *)h_depth);


			std::string exrFilename = outexrpath + std::to_string(m_RGBDAdapter->getFrameNumber()) + ".exr";
			cv::imwrite(exrFilename, imageHDR);
			std::cout << "saved file " << exrFilename << std::endl;

			std::string pngFilename = outpngpath + std::to_string(m_RGBDAdapter->getFrameNumber()) + ".png";
			cv::imwrite(pngFilename, rgbmat);
			std::cout << "saved file" << pngFilename << std::endl;
            */
			delete [] h_depth;
			delete [] h_color;
		}


	private:

		void saveRecordedFramesToFile(const std::string& filename, std::list<float*>& depthFrames) {
			if (depthFrames.size() == 0) return;

			CalibratedSensorData cs;
			cs.m_DepthImageWidth = m_RGBDAdapter->getWidth();
			cs.m_DepthImageHeight = m_RGBDAdapter->getHeight();
			cs.m_ColorImageWidth = 0;
			cs.m_ColorImageHeight = 0;
			cs.m_DepthNumFrames = (unsigned int)depthFrames.size();
			cs.m_ColorNumFrames = 0;
			cs.m_CalibrationDepth.m_Intrinsic = m_RGBDAdapter->getDepthIntrinsics();
			//this is this weird hack of our wrong mat4 intrinsics...
			std::swap(cs.m_CalibrationDepth.m_Intrinsic(0,2), cs.m_CalibrationDepth.m_Intrinsic(0,3));
			std::swap(cs.m_CalibrationDepth.m_Intrinsic(1,2), cs.m_CalibrationDepth.m_Intrinsic(1,3));
			cs.m_CalibrationDepth.m_Intrinsic(2,2) = 1.0f;
			cs.m_CalibrationDepth.m_Intrinsic(1,1) *= -1.0f;
			cs.m_CalibrationDepth.m_Extrinsic.setIdentity();
			cs.m_CalibrationDepth.m_IntrinsicInverse = cs.m_CalibrationDepth.m_Intrinsic.getInverse();
			cs.m_CalibrationDepth.m_ExtrinsicInverse = cs.m_CalibrationDepth.m_Extrinsic.getInverse();
			cs.m_CalibrationColor = cs.m_CalibrationDepth;

			cs.m_DepthImages.resize(cs.m_DepthNumFrames);
			unsigned int dFrame = 0;
			for (auto a : depthFrames) {
				cs.m_DepthImages[dFrame] = a;
				dFrame++;
			}

			std::cout << cs << std::endl;
			std::cout << "dumping recorded frames... ";
			BinaryDataStreamFile outStream(filename, true);
			//BinaryDataStreamZLibFile outStream(filename, true);
			outStream << cs;
			std::cout << "done" << std::endl;

			depthFrames.clear();	//destructor of cs frees all allocated data
		}

		void recordFrame() {
			//TODO MADDI check which image we want to save
			float* h_depth = new float[m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight()];
			cutilSafeCall(cudaMemcpy(h_depth, d_depthMapFilteredFloat, sizeof(float)*m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight(), cudaMemcpyDeviceToHost));
			float* h_depthRefined = new float[m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight()];
			cutilSafeCall(cudaMemcpy(h_depthRefined, d_depthMapRefinedFloat, sizeof(float)*m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight(), cudaMemcpyDeviceToHost));
			

			m_RecordedDepth.push_back(h_depth);
			m_RecordedDepthRefined.push_back(h_depthRefined);

			//std::cout << "frame recorded " << m_RecordedDepthRefined.size() << std::endl;

			m_RGBDAdapter->getRGBDSensor()->recordFrame(); // TODO add back again
		}

		bool isActiveReording() const {
			return m_RecordedDepth.size() > 0 || m_RecordedDepthRefined.size() > 0;
		}

		bool m_bFillHoles;
		bool m_bSFSRefinement;

		bool m_bLightingEstimated;

		CUDAHierarchicalSolverSFS* m_hierarchicalSolverSFS;
		
		CUDAHoleFiller m_HoleFiller;

		mat4f m_rigidTransformIntegrated;
		CUDACameraTrackingMultiRes* m_cameraTracking;
	
		CUDARGBDAdapter* m_RGBDAdapter;

		DX11RGBDRenderer			g_RGBDRenderer;
		DX11CustomRenderTarget		g_CustomRenderTarget;
		

		bool  m_bFilterDepthValues;
		float m_fBilateralFilterSigmaD;
		float m_fBilateralFilterSigmaR;

		bool  m_bFilterIntensityValues;
		float m_fBilateralFilterSigmaDIntensity;
		float m_fBilateralFilterSigmaRIntensity;

		//! depth texture float [D]
		float*			d_depthMapMaskFloat;				// foreground background mask
		float*			d_depthMapMaskMorphFloat;			// foreground background morph
		float*			d_depthMapFilteredFloat;			// float filtered
		float*			d_depthMapFilteredHelperFloat;		// float filtered helper
		float*			d_depthMapRefinedFloat;				// float refined
		float*			d_depthMapRefinedLastFrameFloat;	// float refined
		float4*			d_cameraSpaceFloat4;				// camera space positions

		float*			d_depthMapColorSpaceFloat;			// depth map in camera space
		float*			d_depthMapColorSpaceMorphFloat;		// depth map morphological
		float*			d_depthMapFilteredColorSpaceFloat;	// depth map in camera space

		//! normal texture float [X Y Z *]
		float4*	d_normalMapFloat4;
		float* d_shadingIntensityFloat;

		//! color texture float [R G B A]
		float4* d_colorMapFilteredFloat4;

		//! intensity texture float [I]
		float* d_intensityMapFloat;
		//float* d_hsvMapFloat;

		//! mask of image edges
		unsigned char *d_maskEdgeMapUchar;
		float		  *d_maskEdgeMapFloat; // Only for visualization

		// ! lighting and reflectance
		float* d_lightingCoeffFloat;
		float* d_lightingCoeffLastFrameFloat;
		float4* d_albedoMapFloat4;

		// tracking
		float4* d_cameraSpaceNoRefinementFloat4;
		float4* d_cameraSpaceNoRefinementLastFrameFloat4;
		float4* d_normalMapNoRefinementFloat4;
		float4* d_normalMapNoRefinementLastFrameFloat4;
				
		//! data recording
		std::list<float*> m_RecordedDepth;
		std::list<float*> m_RecordedDepthRefined;

		Timer m_timer;
};

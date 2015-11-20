#include "stdafx.h"

#include "CUDARGBDSensor.h"
#include "TimingLog.h"
#include <algorithm>

extern "C" void convertDepthRawToFloat(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height);
extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height);

extern "C" void convertDepthToColorSpace(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInvs, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void computeShadingValue(float* d_outShading, float* d_indepth, float4* d_normals, float4x4 &Intrinsic, float* d_lighting, unsigned int width, unsigned int height);

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void copyDepthFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void initializeOptimizerMaps(float* d_output, float* d_input, float* d_input2, float* d_mask, unsigned int width, unsigned int height);

extern "C" void setInvalidFloatMap(float* d_output, unsigned int width, unsigned int height);
extern "C" void setInvalidFloat4Map(float4* d_output, unsigned int width, unsigned int height);

extern "C" void computeSimpleSegmentation(float* d_output, float* d_input, float depthThres, unsigned int width, unsigned int height);

extern "C" void computeMaskEdgeMapFloat4(unsigned char* d_output, float4* d_input, float* d_indepth, float threshold,unsigned int width, unsigned int height);

extern "C" void upsampleDepthMap(float* d_output, float* d_input, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);

extern "C" void convertEdgeMaskToFloat(float* d_output, unsigned char* d_input, unsigned int inputWidth, unsigned int inputHeight);

extern "C" void removeDevMeanMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height);

CUDARGBDSensor::CUDARGBDSensor()
{
	m_hierarchicalSolverSFS = NULL;

	m_bFillHoles = false;
	m_bSFSRefinement = true;
	m_bLightingEstimated = false;

	m_bFilterDepthValues = false;
	m_fBilateralFilterSigmaD = 1.0f;
	m_fBilateralFilterSigmaR = 1.0f;

	m_bFilterIntensityValues = false;
	m_fBilateralFilterSigmaDIntensity = 1.0f;
	m_fBilateralFilterSigmaRIntensity = 1.0f;
			
	// depth
	d_depthMapMaskFloat = NULL;
	d_depthMapMaskMorphFloat = NULL;
	d_depthMapRefinedFloat = NULL;
	d_depthMapRefinedLastFrameFloat = NULL;
	d_depthMapFilteredFloat = NULL;
	d_depthMapFilteredHelperFloat = NULL;
	d_cameraSpaceFloat4 = NULL;
		
	d_depthMapColorSpaceFloat = NULL;
	d_depthMapColorSpaceMorphFloat = NULL;

	// color
	d_colorMapFilteredFloat4 = NULL;

	// intensity
	d_intensityMapFloat = NULL;

	// normals
	d_normalMapFloat4 = NULL;

	// mask edge map
	d_maskEdgeMapUchar = NULL;
	d_maskEdgeMapFloat = NULL;

	// tracking
	d_cameraSpaceNoRefinementFloat4 = NULL;
	d_cameraSpaceNoRefinementLastFrameFloat4 = NULL;

	d_normalMapNoRefinementFloat4 = NULL;
	d_normalMapNoRefinementLastFrameFloat4 = NULL;
	
	m_cameraTracking = NULL;

	m_rigidTransformIntegrated.setIdentity();	
}

CUDARGBDSensor::~CUDARGBDSensor()
{
}

void CUDARGBDSensor::OnD3D11DestroyDevice() 
{		
	SAFE_DELETE(m_hierarchicalSolverSFS);

	// depth
	cutilSafeCall(cudaFree(d_depthMapMaskFloat));
	cutilSafeCall(cudaFree(d_depthMapMaskMorphFloat));
	cutilSafeCall(cudaFree(d_depthMapRefinedFloat));
	cutilSafeCall(cudaFree(d_depthMapRefinedLastFrameFloat));
	cutilSafeCall(cudaFree(d_depthMapFilteredFloat));
	cutilSafeCall(cudaFree(d_depthMapFilteredHelperFloat));
	cutilSafeCall(cudaFree(d_cameraSpaceFloat4));

	cutilSafeCall(cudaFree(d_depthMapColorSpaceFloat));
	cutilSafeCall(cudaFree(d_depthMapColorSpaceMorphFloat));
	cutilSafeCall(cudaFree(d_depthMapFilteredColorSpaceFloat));
	
	// color
	cutilSafeCall(cudaFree(d_colorMapFilteredFloat4));

	// intensity
	cutilSafeCall(cudaFree(d_intensityMapFloat));

	// lighting and reflectance
	cutilSafeCall(cudaFree(d_lightingCoeffFloat));
	cutilSafeCall(cudaFree(d_lightingCoeffLastFrameFloat));	
	cutilSafeCall(cudaFree(d_albedoMapFloat4));

	// normals
	cutilSafeCall(cudaFree(d_normalMapFloat4));
	cutilSafeCall(cudaFree(d_shadingIntensityFloat));
	
	// mask edge map
	cutilSafeCall(cudaFree(d_maskEdgeMapUchar));
	cutilSafeCall(cudaFree(d_maskEdgeMapFloat));

	// tracking
	cutilSafeCall(cudaFree(d_cameraSpaceNoRefinementFloat4));
	cutilSafeCall(cudaFree(d_cameraSpaceNoRefinementLastFrameFloat4));
	cutilSafeCall(cudaFree(d_normalMapNoRefinementFloat4));
	cutilSafeCall(cudaFree(d_normalMapNoRefinementLastFrameFloat4));
	
	SAFE_DELETE(m_cameraTracking);	

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();

	reset();
}

HRESULT CUDARGBDSensor::OnD3D11CreateDevice(ID3D11Device* device, CUDARGBDAdapter* CUDARGBDAdapter)
{
	HRESULT hr = S_OK;

	m_RGBDAdapter = CUDARGBDAdapter;	

	assert(GlobalAppState::get().s_optimizationWidthAtLevel[0] == CUDARGBDAdapter->getWidth());
	assert(GlobalAppState::get().s_optimizationHeightAtLevel[0] == CUDARGBDAdapter->getHeight());

	Matrix4f M(m_RGBDAdapter->getColorIntrinsics().ptr()); M.transposeInPlace();	//TODO check!
	m_hierarchicalSolverSFS = new CUDAHierarchicalSolverSFS(M, GlobalAppState::get().s_optimizationWidthAtLevel, GlobalAppState::get().s_optimizationHeightAtLevel, GlobalAppState::get().s_numHierarchyLevels);
	
	const unsigned int bufferDimDepth = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();
	const unsigned int bufferDimColor = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();

	// depth
	cutilSafeCall(cudaMalloc(&d_depthMapMaskFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapMaskMorphFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapRefinedFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapRefinedLastFrameFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapFilteredFloat, sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_depthMapFilteredHelperFloat, sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_cameraSpaceFloat4, 4*sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_cameraSpaceNoRefinementFloat4, 4*sizeof(float)*bufferDimColor));

	cutilSafeCall(cudaMalloc(&d_depthMapColorSpaceFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapColorSpaceMorphFloat, sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_depthMapFilteredColorSpaceFloat, sizeof(float)*bufferDimColor));

	// color
	cutilSafeCall(cudaMalloc(&d_colorMapFilteredFloat4, 4*sizeof(float)*bufferDimColor));

	// intensity
	cutilSafeCall(cudaMalloc(&d_intensityMapFloat, sizeof(float)*bufferDimColor));

	// lighting and reflectance
	cutilSafeCall(cudaMalloc(&d_lightingCoeffFloat, sizeof(float)*9));

	cutilSafeCall(cudaMalloc(&d_lightingCoeffLastFrameFloat, sizeof(float)*9));
	m_hierarchicalSolverSFS->initializeLighting(d_lightingCoeffLastFrameFloat);

	cutilSafeCall(cudaMalloc(&d_albedoMapFloat4, sizeof(float4)*bufferDimColor));

	// normal
	cutilSafeCall(cudaMalloc(&d_normalMapFloat4, 4*sizeof(float)*bufferDimColor));
	cutilSafeCall(cudaMalloc(&d_shadingIntensityFloat, sizeof(float)*bufferDimColor));
			
	// mask edge map
	cutilSafeCall(cudaMalloc(&d_maskEdgeMapUchar, sizeof(unsigned char)*bufferDimColor*2));
	cutilSafeCall(cudaMalloc(&d_maskEdgeMapFloat, sizeof(float)*bufferDimColor));

	// tracking
	cutilSafeCall(cudaMalloc(&d_cameraSpaceNoRefinementFloat4, 4*sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_cameraSpaceNoRefinementLastFrameFloat4, 4*sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_normalMapNoRefinementFloat4, 4*sizeof(float)*bufferDimDepth));
	cutilSafeCall(cudaMalloc(&d_normalMapNoRefinementLastFrameFloat4, 4*sizeof(float)*bufferDimDepth));
	
	m_cameraTracking = new CUDACameraTrackingMultiRes(m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight(), 1);

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(device, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(device, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight, formats));

	return hr;
}


HRESULT CUDARGBDSensor::process(ID3D11DeviceContext* context)
{
	HRESULT hr = S_OK;

	if (m_RGBDAdapter->process(context) == S_FALSE)	return S_FALSE;

	////////////////////////////////////////////////////////////////////////////////////
	// Process Color
	////////////////////////////////////////////////////////////////////////////////////
	
	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if(m_bFilterIntensityValues) gaussFilterFloat4Map(d_colorMapFilteredFloat4, m_RGBDAdapter->GetColorMapResampledFloat4(), m_fBilateralFilterSigmaDIntensity, m_fBilateralFilterSigmaRIntensity, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	else						 copyFloat4Map(d_colorMapFilteredFloat4, m_RGBDAdapter->GetColorMapResampledFloat4(), m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	
	convertColorToIntensityFloat(d_intensityMapFloat, d_colorMapFilteredFloat4, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	
	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeFilterColor += m_timer.getElapsedTimeMS(); TimingLog::countTimeFilterColor++; }

	////////////////////////////////////////////////////////////////////////////////////
	// Process Depth
	////////////////////////////////////////////////////////////////////////////////////

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if(m_bFilterDepthValues) gaussFilterFloatMap(d_depthMapFilteredFloat, m_RGBDAdapter->GetDepthMapResampledFloat(), m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	else					 copyFloatMap(d_depthMapFilteredFloat, m_RGBDAdapter->GetDepthMapResampledFloat(), m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	setInvalidFloatMap(d_depthMapColorSpaceFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeFilterDepth += m_timer.getElapsedTimeMS(); TimingLog::countTimeFilterDepth++; }

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if(GlobalAppState::get().s_bUseCameraCalibration)
	{	
		g_CustomRenderTarget.Clear(context);
		g_CustomRenderTarget.Bind(context);
		g_RGBDRenderer.RenderDepthMap(context, d_depthMapFilteredFloat, d_colorMapFilteredFloat4, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight(), m_RGBDAdapter->getDepthIntrinsicsInv(), m_RGBDAdapter->getDepthExtrinsicsInv(), m_RGBDAdapter->getColorIntrinsics(), g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_remappingDepthDiscontinuityThresOffset, GlobalAppState::get().s_remappingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(context);
		g_CustomRenderTarget.copyToCuda(d_depthMapColorSpaceFloat, 0);
	}
	else
	{
		copyFloatMap(d_depthMapColorSpaceFloat, d_depthMapFilteredFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	}
	
	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeRemapDepth += m_timer.getElapsedTimeMS(); TimingLog::countTimeRemapDepth++; }

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }
	
	if(m_bFillHoles) m_HoleFiller.holeFill(d_depthMapColorSpaceFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeHoleFilling += m_timer.getElapsedTimeMS(); TimingLog::countTimeHoleFilling++; }

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	if(GlobalAppState::get().s_refineForeground) computeSimpleSegmentation(d_depthMapMaskFloat, d_depthMapColorSpaceFloat, GlobalAppState::get().s_segmentationDepthThres, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	else										 copyFloatMap(d_depthMapMaskFloat, d_depthMapColorSpaceFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
		
	copyFloatMap(d_depthMapMaskMorphFloat, d_depthMapMaskFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Initialize Maps for Optimizer !Do this before ICP to only track on foreground
	initializeOptimizerMaps(d_depthMapRefinedFloat, d_depthMapColorSpaceFloat, m_RGBDAdapter->GetDepthMapResampledFloat(), d_depthMapMaskMorphFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeSegment += m_timer.getElapsedTimeMS(); TimingLog::countTimeSegment++; }

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	// For writing out unrefined mesh/and ICP
	float4x4 M((m_RGBDAdapter->getColorIntrinsicsInv()).ptr());
	convertDepthFloatToCameraSpaceFloat4(d_cameraSpaceNoRefinementFloat4, d_depthMapColorSpaceFloat, M, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	
	///////////////////////////////////////////////////////////////////////////////////
	//// Camera Tracking
	///////////////////////////////////////////////////////////////////////////////////	
	
	computeNormals(d_normalMapNoRefinementFloat4, d_cameraSpaceNoRefinementFloat4, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	mat4f deltaTransform = trackRigidTransform(d_cameraSpaceNoRefinementFloat4, d_normalMapNoRefinementFloat4, d_cameraSpaceNoRefinementLastFrameFloat4, d_normalMapNoRefinementLastFrameFloat4, m_RGBDAdapter->getColorIntrinsics());
	
	std::swap(d_cameraSpaceNoRefinementFloat4, d_cameraSpaceNoRefinementLastFrameFloat4);
	std::swap(d_normalMapNoRefinementFloat4, d_normalMapNoRefinementLastFrameFloat4);

	m_rigidTransformIntegrated = m_rigidTransformIntegrated * deltaTransform;

	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeTracking += m_timer.getElapsedTimeMS(); TimingLog::countTimeTracking++; }

	////////////////////////////////////////////////////////////////////////////////////
	// Refine Depth Map
	////////////////////////////////////////////////////////////////////////////////////

	if(m_bSFSRefinement)
	{
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//if(!m_bLightingEstimated)
		{			
			m_hierarchicalSolverSFS->solveLighting(d_depthMapColorSpaceFloat, d_intensityMapFloat, d_normalMapNoRefinementFloat4, d_lightingCoeffLastFrameFloat, d_lightingCoeffFloat, GlobalAppState::get().s_lightingSolverThresDepth);

			cutilSafeCall(cudaMemcpy(d_lightingCoeffLastFrameFloat, d_lightingCoeffFloat, sizeof(float)*9,cudaMemcpyDeviceToDevice));
		
			m_bLightingEstimated = true;
		}

		m_hierarchicalSolverSFS->solveReflectance(d_depthMapColorSpaceFloat, d_colorMapFilteredFloat4, d_normalMapNoRefinementFloat4, d_lightingCoeffFloat, d_albedoMapFloat4);
		computeMaskEdgeMapFloat4(d_maskEdgeMapUchar, d_albedoMapFloat4, d_depthMapColorSpaceFloat, GlobalAppState::get().s_texture_threshold, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
		
		// Stop Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeEstimateLighting += m_timer.getElapsedTimeMS(); TimingLog::countTimeEstimateLighting++; }

		convertEdgeMaskToFloat(d_maskEdgeMapFloat, d_maskEdgeMapUchar, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
		
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		m_hierarchicalSolverSFS->solveSFS(d_depthMapColorSpaceFloat, d_depthMapRefinedLastFrameFloat, d_depthMapMaskMorphFloat, d_intensityMapFloat, d_maskEdgeMapUchar, deltaTransform, GlobalAppState::get().s_nNonLinearIterations, GlobalAppState::get().s_nLinearIterations, GlobalAppState::get().s_nPatchIterations, GlobalAppState::get().s_weightFitting, GlobalAppState::get().s_weightShadingIncrement, GlobalAppState::get().s_weightShadingStart, GlobalAppState::get().s_weightBoundary, GlobalAppState::get().s_weightRegularizer, GlobalAppState::get().s_weightPrior, d_lightingCoeffFloat, NULL /*d_albedoMapFloat4*/, d_depthMapRefinedFloat, GlobalAppState::get().s_refineForeground);
	
		// Stop Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeOptimizer += m_timer.getElapsedTimeMS(); TimingLog::countTimeOptimizer++; }
	}
	else
	{
		m_hierarchicalSolverSFS->initializeLighting(d_lightingCoeffFloat);
		m_hierarchicalSolverSFS->initializeLighting(d_lightingCoeffLastFrameFloat);
		m_bLightingEstimated = false;
	}
	
	// Save last frame
	copyFloatMap(d_depthMapRefinedLastFrameFloat, d_depthMapRefinedFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	
	convertDepthFloatToCameraSpaceFloat4(d_cameraSpaceFloat4, d_depthMapRefinedFloat, M, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	computeNormals(d_normalMapFloat4, d_cameraSpaceFloat4, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());

	float4x4 Mintrinsics((m_RGBDAdapter->getColorIntrinsics()).ptr());
	computeShadingValue(d_shadingIntensityFloat, d_depthMapRefinedFloat, d_normalMapFloat4, Mintrinsics, d_lightingCoeffFloat, m_RGBDAdapter->getWidth(), m_RGBDAdapter->getHeight());
	
	if(GlobalAppState::get().s_bRecordDepthData)
	{
		recordFrame();
	}

	return hr;
}

//! enables bilateral filtering of the depth value
void CUDARGBDSensor::setFiterDepthValues(bool b, float sigmaD, float sigmaR)
{
	m_bFilterDepthValues = b;
	m_fBilateralFilterSigmaD = sigmaD;
	m_fBilateralFilterSigmaR = sigmaR;
}

void CUDARGBDSensor::setFiterIntensityValues(bool b, float sigmaD, float sigmaR)
{
	m_bFilterIntensityValues = b;
	m_fBilateralFilterSigmaDIntensity = sigmaD;
	m_fBilateralFilterSigmaRIntensity = sigmaR;
}

void CUDARGBDSensor::saveCameraSpaceMapAsMesh(bool onlySaveForeground)
{
	saveCameraSpaceMapAsMesh(d_cameraSpaceFloat4, "scan_refined.ply", onlySaveForeground);
	saveCameraSpaceMapAsMesh(d_cameraSpaceNoRefinementLastFrameFloat4, "scan.ply", onlySaveForeground); // because of ICP swap
}

bool CUDARGBDSensor::checkValidT1(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4)
{
	return		(h_cameraSpaceFloat4[y*W+x].z != 0.0 && h_cameraSpaceFloat4[(y+1)*W+x].z != 0.0 && h_cameraSpaceFloat4[y*W+x+1].z != 0.0)
			&&  (h_cameraSpaceFloat4[y*W+x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[(y+1)*W+x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[y*W+x+1].z != -std::numeric_limits<float>::infinity())
			&&  (h_cameraSpaceFloat4[y*W+x].z > GlobalAppState::get().s_minDepth && h_cameraSpaceFloat4[(y+1)*W+x].z > GlobalAppState::get().s_minDepth && h_cameraSpaceFloat4[y*W+x+1].z > GlobalAppState::get().s_minDepth)
			&&  (h_cameraSpaceFloat4[y*W+x].z < GlobalAppState::get().s_maxDepth && h_cameraSpaceFloat4[(y+1)*W+x].z < GlobalAppState::get().s_maxDepth && h_cameraSpaceFloat4[y*W+x+1].z < GlobalAppState::get().s_maxDepth)
			&&  (fabs(h_cameraSpaceFloat4[y*W+x].x) < GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[(y+1)*W+x].x) < GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[y*W+x+1].x) < GlobalAppState::get().s_maxDepth)
			&&  (fabs(h_cameraSpaceFloat4[y*W+x].y) < GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[(y+1)*W+x].y) < GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[y*W+x+1].y) < GlobalAppState::get().s_maxDepth);
}

bool CUDARGBDSensor::checkValidT2(unsigned int x, unsigned int y, unsigned int W, float4* h_cameraSpaceFloat4)
{
	return		(h_cameraSpaceFloat4[(y+1)*W+x].z != 0.0 && h_cameraSpaceFloat4[(y+1)*W+(x+1)].z != 0.0 && h_cameraSpaceFloat4[y*W+x+1].z != 0.0)
			&&  (h_cameraSpaceFloat4[(y+1)*W+x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[(y+1)*W+(x+1)].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[y*W+x+1].z != -std::numeric_limits<float>::infinity())
			&&  (h_cameraSpaceFloat4[(y+1)*W+x].z >  GlobalAppState::get().s_minDepth && h_cameraSpaceFloat4[(y+1)*W+(x+1)].z >  GlobalAppState::get().s_minDepth && h_cameraSpaceFloat4[y*W+x+1].z >  GlobalAppState::get().s_minDepth)
			&&  (h_cameraSpaceFloat4[(y+1)*W+x].z <  GlobalAppState::get().s_maxDepth && h_cameraSpaceFloat4[(y+1)*W+(x+1)].z <  GlobalAppState::get().s_maxDepth && h_cameraSpaceFloat4[y*W+x+1].z <  GlobalAppState::get().s_maxDepth)
			&&  (fabs(h_cameraSpaceFloat4[(y+1)*W+x].x) <  GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[(y+1)*W+(x+1)].x) <  GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[y*W+x+1].x) <  GlobalAppState::get().s_maxDepth)
			&&  (fabs(h_cameraSpaceFloat4[(y+1)*W+x].y) <  GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[(y+1)*W+(x+1)].y) <  GlobalAppState::get().s_maxDepth && fabs(h_cameraSpaceFloat4[y*W+x+1].y) <  GlobalAppState::get().s_maxDepth);
}

void CUDARGBDSensor::saveCameraSpaceMapAsMesh(float4* d_positionData, const std::string& filename, bool onlySaveForeground)
{
	const unsigned int W = m_RGBDAdapter->getWidth();
	const unsigned int H = m_RGBDAdapter->getHeight();

	const unsigned int bufferDimDepth = W*H;
	const unsigned int bufferDimColor = m_RGBDAdapter->getWidth()*m_RGBDAdapter->getHeight();

	float4* h_cameraSpaceFloat4 = new float4[bufferDimColor];
	float4* h_colorMapFilteredFloat4 = new float4[bufferDimColor];
	float*  h_depthMapMask = new float[bufferDimColor];

	cutilSafeCall(cudaMemcpy(h_cameraSpaceFloat4, d_positionData, 4*sizeof(float)*bufferDimColor, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_colorMapFilteredFloat4, d_colorMapFilteredFloat4, 4*sizeof(float)*bufferDimColor, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_depthMapMask, d_depthMapMaskFloat, sizeof(float)*bufferDimColor, cudaMemcpyDeviceToHost));

	std::ofstream out(filename.c_str());

	unsigned int nVertices = W*H;
	
	unsigned int nTriangles = 0;
	for(unsigned int y = 0; y<m_RGBDAdapter->getHeight()-1; y++)
	{
		for(unsigned int x = 0; x<m_RGBDAdapter->getWidth()-1; x++)
		{
			{
				bool isForeground = (h_depthMapMask[y*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[(y+1)*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[y*W+x+1] != -std::numeric_limits<float>::infinity());
				bool isValid	  = checkValidT1(x, y, W, h_cameraSpaceFloat4);

				Vector3f v0(h_cameraSpaceFloat4[y*W+x].x, h_cameraSpaceFloat4[y*W+x].y, h_cameraSpaceFloat4[y*W+x].z);
				Vector3f v1(h_cameraSpaceFloat4[(y+1)*W+x].x, h_cameraSpaceFloat4[(y+1)*W+x].y, h_cameraSpaceFloat4[(y+1)*W+x].z);
				Vector3f v2(h_cameraSpaceFloat4[y*W+x+1].x, h_cameraSpaceFloat4[y*W+x+1].y, h_cameraSpaceFloat4[y*W+x+1].z);
			
				const float dAvg = (1.0f/3.0f)*(v0.z()+v1.z()+v2.z());
				const float edgeThres = GlobalAppState::get().s_edgeThresholdSaveToFileOffset+GlobalAppState::get().s_edgeThresholdSaveToFileLin*dAvg;

				bool hasSmallEdgeLength = ((v0-v1).norm() < edgeThres) && ((v1-v2).norm() < edgeThres) && ((v0-v2).norm() < edgeThres);

				if((isForeground || !onlySaveForeground) && isValid && hasSmallEdgeLength)
				{
					nTriangles++;
				}
			}

			{
				bool isForeground = (h_depthMapMask[(y+1)*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[(y+1)*W+(x+1)] != -std::numeric_limits<float>::infinity() && h_depthMapMask[y*W+x+1] != -std::numeric_limits<float>::infinity());
				bool isValid	  = checkValidT2(x, y, W, h_cameraSpaceFloat4);

				Vector3f v0(h_cameraSpaceFloat4[(y+1)*W+x].x, h_cameraSpaceFloat4[(y+1)*W+x].y, h_cameraSpaceFloat4[(y+1)*W+x].z);
				Vector3f v1(h_cameraSpaceFloat4[(y+1)*W+(x+1)].x, h_cameraSpaceFloat4[(y+1)*W+(x+1)].y, h_cameraSpaceFloat4[(y+1)*W+(x+1)].z);
				Vector3f v2(h_cameraSpaceFloat4[y*W+x+1].x, h_cameraSpaceFloat4[y*W+x+1].y, h_cameraSpaceFloat4[y*W+x+1].z);

				const float dAvg = (1.0f/3.0f)*(v0.z()+v1.z()+v2.z());
				const float edgeThres = GlobalAppState::get().s_edgeThresholdSaveToFileOffset+GlobalAppState::get().s_edgeThresholdSaveToFileLin*dAvg;

				bool hasSmallEdgeLength = ((v0-v1).norm() < edgeThres) && ((v1-v2).norm() < edgeThres) && ((v0-v2).norm() < edgeThres);

				if((isForeground || !onlySaveForeground) && isValid && hasSmallEdgeLength)
				{
					nTriangles++;
				}
			}
		}
	}
	
	if(out.fail()) { std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " __FUNCTION__ << std::endl; return; }

	//write header
	out << 
		"ply\n" <<
		"format ascii 1.0\n" <<
		"element vertex " << nVertices << "\n" <<
		"property float x\n" <<
		"property float y\n" <<
		"property float z\n" <<
		"property uchar red\n" <<
		"property uchar green\n" <<
		"property uchar blue\n" <<
		"element face " << nTriangles << "\n" <<
		"property list uchar int vertex_index\n" <<
		"end_header\n";

	for(unsigned int y = 0; y<m_RGBDAdapter->getHeight(); y++)
	{
		for(unsigned int x = 0; x<m_RGBDAdapter->getWidth(); x++)
		{
			float4 p = h_cameraSpaceFloat4[y*W+x];
			float4 c = h_colorMapFilteredFloat4[y*W+x];

			if(h_cameraSpaceFloat4[y*W+x].z != -std::numeric_limits<float>::infinity() && h_cameraSpaceFloat4[y*W+x].z != 0.0f) out << p.x << " " << p.y << " " << p.z << " " << (int)(255.0f*c.x) << " " << (int)(255.0f*c.y) << " " << (int)(255.0f*c.z) << std::endl;
			else																												out << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << std::endl;
		}
	}

	for(unsigned int y = 0; y<m_RGBDAdapter->getHeight()-1; y++)
	{
		for(unsigned int x = 0; x<m_RGBDAdapter->getWidth()-1; x++)
		{
			{
				bool isForeground = (h_depthMapMask[y*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[(y+1)*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[y*W+x+1] != -std::numeric_limits<float>::infinity());
				bool isValid	  = checkValidT1(x, y, W, h_cameraSpaceFloat4);

				Vector3f v0(h_cameraSpaceFloat4[y*W+x].x, h_cameraSpaceFloat4[y*W+x].y, h_cameraSpaceFloat4[y*W+x].z);
				Vector3f v1(h_cameraSpaceFloat4[(y+1)*W+x].x, h_cameraSpaceFloat4[(y+1)*W+x].y, h_cameraSpaceFloat4[(y+1)*W+x].z);
				Vector3f v2(h_cameraSpaceFloat4[y*W+x+1].x, h_cameraSpaceFloat4[y*W+x+1].y, h_cameraSpaceFloat4[y*W+x+1].z);
			
				const float dAvg = (1.0f/3.0f)*(v0.z()+v1.z()+v2.z());
				const float edgeThres = GlobalAppState::get().s_edgeThresholdSaveToFileOffset+GlobalAppState::get().s_edgeThresholdSaveToFileLin*dAvg;

				bool hasSmallEdgeLength = ((v0-v1).norm() < edgeThres) && ((v1-v2).norm() < edgeThres) && ((v0-v2).norm() < edgeThres);

				if((isForeground || !onlySaveForeground) && isValid && hasSmallEdgeLength)
				{
					out << 3 << " " << y*W+x << " " << y*W+(x+1) << " " << (y+1)*W+x << std::endl;
				}
			}

			{
				bool isForeground = (h_depthMapMask[(y+1)*W+x] != -std::numeric_limits<float>::infinity() && h_depthMapMask[(y+1)*W+(x+1)] != -std::numeric_limits<float>::infinity() && h_depthMapMask[y*W+x+1] != -std::numeric_limits<float>::infinity());
				bool isValid	  = checkValidT2(x, y, W, h_cameraSpaceFloat4);

				Vector3f v0(h_cameraSpaceFloat4[(y+1)*W+x].x, h_cameraSpaceFloat4[(y+1)*W+x].y, h_cameraSpaceFloat4[(y+1)*W+x].z);
				Vector3f v1(h_cameraSpaceFloat4[(y+1)*W+(x+1)].x, h_cameraSpaceFloat4[(y+1)*W+(x+1)].y, h_cameraSpaceFloat4[(y+1)*W+(x+1)].z);
				Vector3f v2(h_cameraSpaceFloat4[y*W+x+1].x, h_cameraSpaceFloat4[y*W+x+1].y, h_cameraSpaceFloat4[y*W+x+1].z);

				const float dAvg = (1.0f/3.0f)*(v0.z()+v1.z()+v2.z());
				const float edgeThres = GlobalAppState::get().s_edgeThresholdSaveToFileOffset+GlobalAppState::get().s_edgeThresholdSaveToFileLin*dAvg;

				bool hasSmallEdgeLength = ((v0-v1).norm() < edgeThres) && ((v1-v2).norm() < edgeThres) && ((v0-v2).norm() < edgeThres);

				if((isForeground || !onlySaveForeground) && isValid && hasSmallEdgeLength)
				{
					out << 3 << " " << (y+1)*W+x << " " << y*W+x+1 << " " << (y+1)*W+(x+1) << std::endl;
				}
			}
		}
	}

	out.close();

	std::cout << "Dumping of " << filename << " finished" << std::endl;

	delete [] h_cameraSpaceFloat4;
	delete [] h_colorMapFilteredFloat4;
}

float* CUDARGBDSensor::GetFilteredDepthMapMaskFloat()
{
	return d_depthMapMaskFloat;
}

float4* CUDARGBDSensor::GetCameraSpacePositionsFloat4()
{
	return d_cameraSpaceFloat4;
}

float* CUDARGBDSensor::GetDepthMapRefinedFloat()
{
	return d_depthMapRefinedFloat;
}

float* CUDARGBDSensor::GetDepthMapRefinedLastFrameFloat()
{
	return d_depthMapRefinedLastFrameFloat;
}

float* CUDARGBDSensor::GetDepthMapColorSpaceFloat()
{
	return d_depthMapColorSpaceFloat;
}

float4* CUDARGBDSensor::GetAlbedoFloat4()
{
	return d_albedoMapFloat4;
}

float* CUDARGBDSensor::GetShadingFloat()
{
	return d_shadingIntensityFloat;
}

float4* CUDARGBDSensor::GetColorMapFilteredFloat4()
{
	return d_colorMapFilteredFloat4;
}

float* CUDARGBDSensor::GetIntensityMapFloat()
{
	return d_intensityMapFloat;
}

float4* CUDARGBDSensor::GetNormalMapFloat4()
{
	return d_normalMapFloat4;
}

float* CUDARGBDSensor::GetEdgeMask()
{
	return d_maskEdgeMapFloat;
}


unsigned int CUDARGBDSensor::getDepthWidth() const
{
	return m_RGBDAdapter->getWidth();
}

unsigned int CUDARGBDSensor::getDepthHeight() const
{
	return m_RGBDAdapter->getHeight();
}

unsigned int CUDARGBDSensor::getColorWidth() const
{
	return m_RGBDAdapter->getWidth();
}

unsigned int CUDARGBDSensor::getColorHeight() const
{
	return m_RGBDAdapter->getHeight();
}

mat4f CUDARGBDSensor::trackRigidTransform(float4* d_positionsCurrent, float4* d_normalsCurrent, float4* d_positionsLast, float4* d_normalsLast, const mat4f& intrinsics)
{
	mat4f trans = mat4f::identity();
	if(m_RGBDAdapter->getFrameNumber() > 10)
	{
		ICPErrorLog log;
		mat4f transform = mat4f::identity();
		trans = m_cameraTracking->applyCT(
			d_positionsCurrent, d_normalsCurrent, NULL,
			d_positionsLast, d_normalsLast, NULL,
			transform,
			GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
			GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
			100.0f, 3.0f,
			mat4f::identity(),
			GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
			intrinsics, NULL);
		//log.printErrorLastFrame();
		//std::cout << trans << std::endl;
	}

	return trans;
}

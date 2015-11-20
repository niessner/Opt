#include "stdafx.h"

#include "CUDACameraTrackingMultiRes.h"
#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include <iostream>
#include <limits>

/////////////////////////////////////////////////////
// Camera Tracking Multi Res
/////////////////////////////////////////////////////

CUDACameraTrackingMultiRes::CUDACameraTrackingMultiRes(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels) {
	assert(levels == 1); //implement rest...
	m_levels = levels;

	d_correspondence = new float4*[m_levels];
	d_correspondenceNormal = new float4*[m_levels];

	d_input = new float4*[m_levels];
	d_inputNormal = new float4*[m_levels];
	d_inputColor = new float4*[m_levels];

	d_model = new float4*[m_levels];
	d_modelNormal = new float4*[m_levels];
	d_modelColor = new float4*[m_levels];

	m_imageWidth = new unsigned int[m_levels];
	m_imageHeight = new unsigned int[m_levels];

	

	unsigned int fac = 1;
	for (unsigned int i = 0; i<GlobalCameraTrackingState::getInstance().s_maxLevels; i++) {
		m_imageWidth[i] = imageWidth/fac;
		m_imageHeight[i] = imageHeight/fac;
		
		// correspondences
		cutilSafeCall(cudaMalloc(&d_correspondence[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_correspondenceNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));

		// input
		if (i != 0) {  // Not finest level
			cutilSafeCall(cudaMalloc(&d_input[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_inputNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_inputColor[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		} else {
			d_input[i] = NULL;
			d_inputNormal[i] = NULL;
			d_inputColor[i] = NULL;
		}

		// model
		if (i != 0) { // Not fines level
			cutilSafeCall(cudaMalloc(&d_model[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_modelNormal[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_modelColor[i], sizeof(float4)*m_imageWidth[i]*m_imageHeight[i]));
		} else {
			d_model[i] = NULL;
			d_modelNormal[i] = NULL;
			d_modelColor[i] = NULL;
		}
		fac*=2;
	}

	m_matrixTrackingLost.fill(-std::numeric_limits<float>::infinity());

	m_CUDABuildLinearSystem = new CUDABuildLinearSystem(m_imageWidth[0], m_imageHeight[0]);
}

CUDACameraTrackingMultiRes::~CUDACameraTrackingMultiRes() {

	d_input[0] = NULL;
	d_inputNormal[0] = NULL;
	d_inputColor[0] = NULL;

	d_model[0] = NULL;
	d_modelNormal[0] = NULL;
	d_modelColor[0] = NULL;

	// correspondence
	if (d_correspondence) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_correspondence[i])	cutilSafeCall(cudaFree(d_correspondence[i]));
		SAFE_DELETE_ARRAY(d_correspondence)
	}
	if (d_correspondenceNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_correspondenceNormal[i])	cutilSafeCall(cudaFree(d_correspondenceNormal[i]));
		SAFE_DELETE_ARRAY(d_correspondenceNormal)
	}

	// input
	if (d_input) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_input[i])	cutilSafeCall(cudaFree(d_input[i]));
		SAFE_DELETE_ARRAY(d_input)
	}
	if (d_inputNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_inputNormal[i])	cutilSafeCall(cudaFree(d_inputNormal[i]));
		SAFE_DELETE_ARRAY(d_inputNormal)
	}
	if (d_inputColor) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_inputColor[i])	cutilSafeCall(cudaFree(d_inputColor[i]));
		SAFE_DELETE_ARRAY(d_inputColor)
	}

	// model
	if (d_model) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_model[i])	cutilSafeCall(cudaFree(d_model[i]));
		SAFE_DELETE_ARRAY(d_model)
	}
	if (d_modelNormal) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelNormal[i])	cutilSafeCall(cudaFree(d_modelNormal[i]));
		SAFE_DELETE_ARRAY(d_modelNormal)
	}
	if (d_modelColor) {
		for (unsigned int i = 0; i < m_levels; i++) 
			if (d_modelColor[i])	cutilSafeCall(cudaFree(d_modelColor[i]));
		SAFE_DELETE_ARRAY(d_modelColor)
	}

	if (m_imageWidth)	SAFE_DELETE_ARRAY(m_imageWidth);
	if (m_imageHeight)	SAFE_DELETE_ARRAY(m_imageHeight);

	SAFE_DELETE(m_CUDABuildLinearSystem);
}

bool CUDACameraTrackingMultiRes::checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres) {
	Eigen::AngleAxisf aa(R);

	if (aa.angle() > angleThres || t.norm() > distThres) {
		std::cout << "Tracking lost: angle " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	//std::cout << "Tracking successful: anlge " << (aa.angle()/M_PI)*180.0f << " translation " << t.norm() << std::endl;
	return true;
}

Eigen::Matrix4f CUDACameraTrackingMultiRes::delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level) 
{
	Eigen::Matrix3f R =	 Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()  // Rot Z
		*Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()  // Rot Y
		*Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix(); // Rot X

	Eigen::Vector3f t = x.segment(3, 3);

	if(!checkRigidTransformation(R, t, GlobalCameraTrackingState::getInstance().s_angleTransThres[level], GlobalCameraTrackingState::getInstance().s_distTransThres[level])) {
		return m_matrixTrackingLost;
	}

	Eigen::Matrix4f res; res.setIdentity();
	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = meanStDev*t+mean-R*mean;

	return res;
}

Eigen::Matrix4f CUDACameraTrackingMultiRes::computeBestRigidAlignment(float4* dInput, float4* dInputNormals, float3& mean, float meanStDev, float nValidCorres, const Eigen::Matrix4f& globalDeltaTransform, unsigned int level, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf)
{
	Eigen::Matrix4f deltaTransform = globalDeltaTransform;

	for (unsigned int i = 0; i < maxInnerIter; i++)
	{
		conf.reset();

		Matrix6x7f system;

		m_CUDABuildLinearSystem->applyBL(dInput, d_correspondence[level], d_correspondenceNormal[level], mean, meanStDev, deltaTransform, m_imageWidth[level], m_imageHeight[level], level, system, conf);
		 
		Matrix6x6f ATA = system.block(0, 0, 6, 6);
		Vector6f ATb = system.block(0, 6, 6, 1);

		if (ATA.isZero()) {
			return m_matrixTrackingLost;
		}

		Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vector6f x = SVD.solve(ATb);

		//computing the matrix condition
		Vector6f evs = SVD.singularValues();
		conf.matrixCondition = evs[0]/evs[5];

		Eigen::Matrix4f t = delinearizeTransformation(x, Eigen::Vector3f(mean.x, mean.y, mean.z), meanStDev, level);
		if(t(0, 0) == -std::numeric_limits<float>::infinity())
		{
			conf.trackingLostTresh = true;
			return m_matrixTrackingLost;
		}

		deltaTransform = t*deltaTransform;
	}
	return deltaTransform;
}

mat4f CUDACameraTrackingMultiRes::applyCT(
	float4* dInput, float4* dInputNormals, float4* dInputColors, 
	float4* dModel, float4* dModelNormals, float4* dModelColors, 
	const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter, 
	const std::vector<float>& distThres, const std::vector<float>& normalThres, float condThres, float angleThres, 
	const mat4f& deltaTransformEstimate, const std::vector<float>& earlyOutResidual, 
	const mat4f& intrinsic,
	ICPErrorLog* errorLog)
{		
	// Input
	d_input[0] = dInput;
	d_inputNormal[0] = dInputNormals;
	d_inputColor[0] = dInputColors;

	d_model[0] = dModel;
	d_modelNormal[0] = dModelNormals;
	d_modelColor[0] = dModelColors;

	for (unsigned int i = 0; i < m_levels-1; i++)
	{
		assert(false);
		MLIB_EXCEPTION("not supported yet");
		// Downsample Depth Maps directly ? -> better ?
		//DX11ImageHelper::applyDownsampling(context, m_inputTextureFloat4SRV[i], m_inputTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		//DX11ImageHelper::applyDownsampling(context, m_modelTextureFloat4SRV[i], m_modelTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);

		//DX11ImageHelper::applyNormalComputation(context, m_inputTextureFloat4SRV[i+1], m_inputNormalTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		//DX11ImageHelper::applyNormalComputation(context, m_modelTextureFloat4SRV[i+1], m_modelNormalTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);

		//DX11ImageHelper::applyDownsampling(context, m_inputColorTextureFloat4SRV[i], m_inputColorTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
		//DX11ImageHelper::applyDownsampling(context, m_modelColorTextureFloat4SRV[i], m_modelColorTextureFloat4UAV[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
	}

	Eigen::Matrix4f deltaTransform; 
	//deltaTransform.setIdentity();
	deltaTransform = MatToEig(deltaTransformEstimate);
	for (int level = GlobalCameraTrackingState::getInstance().s_maxLevels-1; level>=0; level--)	{	
		if (errorLog) {
			errorLog->newICPFrame(level);
		}

		deltaTransform = align(d_input[level], d_inputNormal[level], d_inputColor[level], d_model[level], d_modelNormal[level], d_modelColor[level], deltaTransform, level, maxInnerIter[level], maxOuterIter[level], distThres[level], normalThres[level], condThres, angleThres, earlyOutResidual[level], intrinsic, errorLog);

		if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity()) {
			return EigToMat(m_matrixTrackingLost);
		}
	}


	return lastTransform*EigToMat(deltaTransform);
}

Eigen::Matrix4f CUDACameraTrackingMultiRes::align(float4* dInput, float4* dInputNormals, float4* dInputColors, float4* dModel, float4* dModelNormals, float4* dModelColors, Eigen::Matrix4f& deltaTransform, unsigned int level, unsigned int maxInnerIter, unsigned maxOuterIter, float distThres, float normalThres, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, ICPErrorLog* errorLog)
{
	float lastICPError = -1.0f;
	for(unsigned int i = 0; i<maxOuterIter; i++)
	{
		float3 mean;
		float meanStDev;
		float nValidCorres;

		LinearSystemConfidence currConfWiReject;
		LinearSystemConfidence currConfNoReject;

		if (errorLog) {
			//run ICP without correspondence rejection (must be run before because it needs the old delta transform)
			float dThresh = 1000.0f;	float nThresh = 0.0f;
			computeCorrespondences(dInput, dInputNormals, dInputColors, dModel, dModelNormals, dModelColors, mean, meanStDev, nValidCorres, deltaTransform, level, dThresh, nThresh, intrinsic);
			computeBestRigidAlignment(dInput, dInputNormals, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfNoReject);
			errorLog->addCurrentICPIteration(currConfNoReject, level);
		}

		//standard correspondence search and alignment
		computeCorrespondences(dInput, dInputNormals, dInputColors, dModel, dModelNormals, dModelColors, mean, meanStDev, nValidCorres, deltaTransform, level, distThres, normalThres, intrinsic);
		deltaTransform = computeBestRigidAlignment(dInput, dInputNormals, mean, meanStDev, nValidCorres, deltaTransform, level, maxInnerIter, condThres, angleThres, currConfWiReject);

		if (std::abs(lastICPError - currConfWiReject.sumRegError) < earlyOut) {
			//std::cout << "ICP aboarted because no further convergence... " << i << std::endl;
			break;
		}
		lastICPError = currConfWiReject.sumRegError;
	}

	return deltaTransform;
}

void CUDACameraTrackingMultiRes::computeCorrespondences(float4* dInput, float4* dInputNormals, float4* dInputColors, float4* dModel, float4* dModelNormals, float4* dModelColors, float3& mean, float& meanStDev, float& nValidCorres, const Eigen::Matrix4f& deltaTransform, unsigned int level, float distThres, float normalThres, const mat4f& intrinsic)
{
	float levelFactor = pow(2.0f, (float)level);
	
	mean = make_float3(0.0f, 0.0f, 0.0f);
	meanStDev = 1.0f;
	CUDAImageHelper::applyProjectiveCorrespondences(
		dInput, dInputNormals, NULL, 
		dModel, dModelNormals, NULL, 
		d_correspondence[level], d_correspondenceNormal[level], deltaTransform, m_imageWidth[level], m_imageHeight[level], distThres, normalThres, levelFactor, intrinsic
		);
}

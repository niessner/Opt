#pragma once

/************************************************************************/
/* Main ICP class used for tracking                                     */
/************************************************************************/

#include "MatrixConversion.h"
#include "DX11QuadDrawer.h"
#include "CUDABuildLinearSystem.h"
#include "ICPErrorLog.h"
#include "Eigen.h"

#include <D3D11.h>
#include "DX11Utils.h"


using namespace MatrixConversion;


class CUDACameraTrackingMultiRes
{
public:
	CUDACameraTrackingMultiRes(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels);
	~CUDACameraTrackingMultiRes();

	mat4f applyCT(
		float4* dInput, float4* dInputNormals, float4* dInputColors, 
		float4* dModel, float4* dModelNormals, float4* dModelColors, 
		const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter, 
		const std::vector<float>& distThres, const std::vector<float>& normalThres, float condThres, float angleThres, 
		const mat4f& deltaTransformEstimate, const std::vector<float>& earlyOutResidual, 
		const mat4f& intrinsic,
		ICPErrorLog* errorLog);

private:

	// angleThres in radians, distThres in meter
	bool checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres);

	Eigen::Matrix4f delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level);

	Eigen::Matrix4f computeBestRigidAlignment(float4* dInput, float4* dInputNormals, float3& mean, float meanStDev, float nValidCorres, const Eigen::Matrix4f& globalDeltaTransform, unsigned int level, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf);

	Eigen::Matrix4f align(float4* dInput, float4* dInputNormals, float4* dInputColors, float4* dModel, float4* dModelNormals, float4* dModelColors, Eigen::Matrix4f& deltaTransform, unsigned int level, unsigned int maxInnerIter, unsigned maxOuterIter, float distThres, float normalThres, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, ICPErrorLog* errorLog);

	void computeCorrespondences(float4* dInput, float4* dInputNormals, float4* dInputColors, float4* dModel, float4* dModelNormals, float4* dModelColors, float3& mean, float& meanStDev, float& nValidCorres, const Eigen::Matrix4f& deltaTransform, unsigned int level, float distThres, float normalThres, const mat4f& intrinsic);
	

	float4** d_correspondence;
	float4** d_correspondenceNormal;

	float4** d_input;
	float4** d_inputNormal;
	float4** d_inputColor;

	float4** d_model;
	float4** d_modelNormal;
	float4** d_modelColor;

	// Image Pyramid Dimensions
	unsigned int* m_imageWidth;
	unsigned int* m_imageHeight;
	unsigned int m_levels;

	Eigen::Matrix4f m_matrixTrackingLost;

	CUDABuildLinearSystem* m_CUDABuildLinearSystem;
};

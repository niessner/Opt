#include <cutil_inline.h>
#include <cutil_math.h>
inline __device__ mat3x3 evalRMat(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma) {
	mat3x3 R;
	R(0, 0) = CosGamma*CosBeta;
	R(0, 1) = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R(0, 2) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R(1, 0) = SinGamma*CosBeta;
	R(1, 1) = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	R(1, 2) = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R(2, 0) = -SinBeta;
	R(2, 1) = CosBeta*SinAlpha;
	R(2, 2) = CosBeta*CosAlpha;
	return R;
}
inline __device__ mat3x3 evalR(const float3& angles) {
	return evalRMat(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}
__inline__ __device__ float evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters) {
	float3 e = make_float3(0.0f, 0.0f, 0.0f);
	if (state.d_target[variableIdx].x != MINF) {
		float3 e_fit = state.d_x[variableIdx] - state.d_target[variableIdx];
		e += parameters.weightFitting*e_fit*e_fit;
	}
	float3	 e_reg = make_float3(0.0f, 0.0f, 0.0F);
	float3x3 R = evalR(state.d_a[variableIdx]);
	float3   p = state.d_x[variableIdx];
	float3   pHat = state.d_urshape[variableIdx];
	int numNeighbours = input.d_numNeighbours[variableIdx];
	for (unsigned int i = 0; i < numNeighbours; i++) {
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		float3 q = state.d_x[neighbourIndex];
		float3 qHat = state.d_urshape[neighbourIndex];
		float3 d = (p - q) - R*(pHat - qHat);
		e_reg += d*d;
	}
	e += parameters.weightRegularizer*e_reg;
	float res = e.x + e.y + e.z;
	return res;
}
inline __device__ mat3x3 evalDerivativeRotationTimesVector(const float3x3& dRAlpha, const float3x3& dRBeta, const float3x3& dRGamma, const float3& d){
	mat3x3 R; 
	float3 b = dRAlpha*d; 
	R(0, 0) = b.x; R(1, 0) = b.y; R(2, 0) = b.z;
	b = dRBeta *d;
	R(0, 1) = b.x; R(1, 1) = b.y; R(2, 1) = b.z;
	b = dRGamma*d; 
	R(0, 2) = b.x; R(1, 2) = b.y; R(2, 2) = b.z;
	return R;
}
inline __device__ mat3x3 evalRMat_dAlpha(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma) {
	mat3x3 R;
	R(0, 0) = 0.0f;
	R(0, 1) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R(0, 2) = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;
	R(1, 0) = 0.0f;
	R(1, 1) = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R(1, 2) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	R(2, 0) = 0.0f;
	R(2, 1) = CosBeta*CosAlpha;
	R(2, 2) = -CosBeta*SinAlpha;
	return R;
}
inline __device__ mat3x3 evalRMat_dBeta(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma) {
	mat3x3 R;
	R(0, 0) = -CosGamma*SinBeta;
	R(0, 1) = CosGamma*CosBeta*SinAlpha;
	R(0, 2) = CosGamma*CosBeta*CosAlpha;
	R(1, 0) = -SinGamma*SinBeta;
	R(1, 1) = SinGamma*CosBeta*SinAlpha;
	R(1, 2) = SinGamma*CosBeta*CosAlpha;
	R(2, 0) = -CosBeta;
	R(2, 1) = -SinBeta*SinAlpha;
	R(2, 2) = -SinBeta*CosAlpha;
	return R;
}
inline __device__ mat3x3 evalRMat_dGamma(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma) {
	mat3x3 R;
	R(0, 0) = -SinGamma*CosBeta;
	R(0, 1) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	R(0, 2) = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha;
	R(1, 0) = CosGamma*CosBeta;
	R(1, 1) = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R(1, 2) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R(2, 0) = 0.0f;
	R(2, 1) = 0.0f;
	R(2, 2) = 0.0f;
	return R;
}
inline __device__ void evalDerivativeRotationMatrix(const float3& angles, mat3x3& dRAlpha, mat3x3& dRBeta, mat3x3& dRGamma) {
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);
	dRAlpha = evalRMat_dAlpha(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRBeta = evalRMat_dBeta(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRGamma = evalRMat_dGamma(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}
__inline__ __device__ float3 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& outAngle) {
	mat3x1 ones; ones(0) = 1.0f; ones(1) = 1.0f; ones(2) = 1.0f;
	state.d_delta [variableIdx]	= make_float3(0.0f, 0.0f, 0.0f);
	state.d_deltaA[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);
	mat3x1 b;  b.setZero();
	mat3x1 bA; bA.setZero();
	mat3x1 pre; pre.setZero();
	mat3x3 preA; preA.setZero();
	mat3x1 p = mat3x1(state.d_x[variableIdx]);
	mat3x1 t = mat3x1(state.d_target[variableIdx]);
	if (state.d_target[variableIdx].x != MINF) {
		b   -= 2.0f*parameters.weightFitting * (p - t);
		pre += 2.0f*parameters.weightFitting * ones;
	}
	mat3x1 e_reg; e_reg.setZero();
	mat3x1 e_reg_angle; e_reg_angle.setZero();
	mat3x1 pHat = mat3x1(state.d_urshape[variableIdx]);
	mat3x3 R_i = evalR(state.d_a[variableIdx]);
	mat3x3 dRAlpha, dRBeta, dRGamma;
	evalDerivativeRotationMatrix(state.d_a[variableIdx], dRAlpha, dRBeta, dRGamma);
	int numNeighbours = input.d_numNeighbours[variableIdx];
	for (unsigned int i = 0; i < numNeighbours; i++) {
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		mat3x1 q	= mat3x1(state.d_x[neighbourIndex]);
		mat3x1 qHat = mat3x1(state.d_urshape[neighbourIndex]);
		mat3x3 R_j  = evalR(state.d_a[neighbourIndex]);
		mat3x3 D    = -evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, pHat - qHat);
		mat3x3 P	= parameters.weightRegularizer*D.getTranspose()*D;
		e_reg		+= 2.0f*(p - q) - (R_i+R_j)*(pHat - qHat);
		pre			+= 2.0f*(2.0f*parameters.weightRegularizer*ones);
		e_reg_angle += D.getTranspose()*((p - q) - R_i*(pHat - qHat));
		preA		+= 2.0f*P;
	}
	b  += -2.0f*parameters.weightRegularizer*e_reg;
	bA += -2.0f*parameters.weightRegularizer*e_reg_angle;
	if (fabs(pre(0)) > FLOAT_EPSILON && fabs(pre(1)) > FLOAT_EPSILON && fabs(pre(2)) > FLOAT_EPSILON) { pre(0) = 1.0f/pre(0);  pre(1) = 1.0f/pre(1);  pre(2) = 1.0f/pre(2); } else { pre = ones; }
	state.d_precondioner[variableIdx] = make_float3(pre(0), pre(1), pre(2));
	if (preA(0, 0) > FLOAT_EPSILON) {
		preA(0, 0) = 1.0f / preA(0, 0); 
		preA(1, 1) = 1.0f / preA(1, 1);
		preA(2, 2) = 1.0f / preA(2, 2);
	} else { 
		preA(0, 0) = 1.0f;
		preA(1, 1) = 1.0f;
		preA(2, 2) = 1.0f;
	}
	state.d_precondionerA[variableIdx] = make_float3(preA(0, 0), preA(1, 1), preA(2, 2));
	outAngle = bA;
	return b;
}
__inline__ __device__ float3 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& outAngle)
{
	mat3x1 b;  b.setZero();
	mat3x1 bA; bA.setZero();
	mat3x1 p = mat3x1(state.d_p[variableIdx]);
	if (state.d_target[variableIdx].x != MINF)
		b += 2.0f*parameters.weightFitting*p;
	mat3x1	e_reg; e_reg.setZero();
	mat3x1	e_reg_angle; e_reg_angle.setZero();
	mat3x3 dRAlpha, dRBeta, dRGamma;
	evalDerivativeRotationMatrix(state.d_a[variableIdx], dRAlpha, dRBeta, dRGamma);
	mat3x1 pHat = mat3x1(state.d_urshape[variableIdx]);
	mat3x1 pAngle = mat3x1(state.d_pA[variableIdx]);
	int numNeighbours = input.d_numNeighbours[variableIdx];
	for (unsigned int i = 0; i < numNeighbours; i++) {
		unsigned int neighbourIndex = input.d_neighbourIdx[input.d_neighbourOffset[variableIdx] + i];
		mat3x1 qHat = mat3x1(state.d_urshape[neighbourIndex]);
		mat3x3 D	= -evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, pHat - qHat);
		mat3x3 dRAlphaJ, dRBetaJ, dRGammaJ;
		evalDerivativeRotationMatrix(state.d_a[neighbourIndex], dRAlphaJ, dRBetaJ, dRGammaJ);
		mat3x3 D_j = -evalDerivativeRotationTimesVector(dRAlphaJ, dRBetaJ, dRGammaJ, pHat - qHat);
		mat3x1 q = mat3x1(state.d_p[neighbourIndex]);
		mat3x1 qAngle = mat3x1(state.d_pA[neighbourIndex]);
		e_reg		+= 2.0f*(p-q);
		e_reg_angle += D.getTranspose()*D*pAngle;
		e_reg		+= D*pAngle + D_j*qAngle;
		e_reg_angle += D.getTranspose()*(p - q);
	}
	b  += 2.0f*parameters.weightRegularizer*e_reg;
	bA += 2.0f*parameters.weightRegularizer*e_reg_angle;
	outAngle = bA;
	return b;
}

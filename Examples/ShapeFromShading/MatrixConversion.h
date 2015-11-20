#pragma once

#include "Eigen.h"
#include "d3dx9math.h"

namespace MatrixConversion
{
	//static D3DXMATRIX EigToMat(const Eigen::Matrix4f& mat)
	//{
	//	D3DXMATRIX m(mat.data());
	//	D3DXMATRIX res; D3DXMatrixTranspose(&res, &m);

	//	return res;
	//}
	//static Eigen::Matrix4f MatToEig(const D3DXMATRIX& mat)
	//{
	//	return Eigen::Matrix4f((float*)mat.m).transpose();
	//}

	static mat4f EigToMat(const Eigen::Matrix4f& mat)
	{
		return mat4f(mat.data()).getTranspose();
	}

	static Eigen::Matrix4f MatToEig(const mat4f& mat)
	{
		return Eigen::Matrix4f(mat.ptr()).transpose();
	}

	static Eigen::Vector4f VecH(const Eigen::Vector3f& v)
	{
		return Eigen::Vector4f(v[0], v[1], v[2], 1.0);
	}

	static Eigen::Vector3f VecDH(const Eigen::Vector4f& v)
	{
		return Eigen::Vector3f(v[0]/v[3], v[1]/v[3], v[2]/v[3]);
	}

	static Eigen::Vector3f VecToEig(const vec3f& v)
	{
		return Eigen::Vector3f(v[0], v[1], v[2]);
	}

	static vec3f EigToVec(const Eigen::Vector3f& v)
	{
		return vec3f(v[0], v[1], v[2]);
	}

	static vec3f EigToVec(const D3DXMATRIX v)
	{
		return vec3f(v[0], v[1], v[2]);
	}


	// dx conversion
	static vec3f toMlib(const D3DXVECTOR3& v) {
		return vec3f(v.x, v.y, v.z);
	}
	static vec4f toMlib(const D3DXVECTOR4& v) {
		return vec4f(v.x, v.y, v.z, v.w);
	}
	static mat4f toMlib(const D3DXMATRIX& m) {
		mat4f c((const float*)&m);
		return c.getTranspose();
	}
	static D3DXVECTOR3 toDX(const vec3f& v) {
		return D3DXVECTOR3(v.x, v.y, v.z);
	}
	static D3DXVECTOR4 toDX(const vec4f& v) {
		return D3DXVECTOR4(v.x, v.y, v.z, v.w);
	}
	static D3DXMATRIX toDX(const mat4f& m) {
		D3DXMATRIX c((const float*)m.ptr());
		D3DXMatrixTranspose(&c, &c);
		return c;
	}

}

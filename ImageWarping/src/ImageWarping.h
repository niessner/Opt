#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32& image, std::vector<std::vector<int>>& constraints) {
		m_image = image;
	
		cutilSafeCall(cudaMalloc(&d_urshape,    sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpField,  sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpAngles, sizeof(float3)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_constraints, sizeof(float2)*m_image.getWidth()*m_image.getHeight()));

		float2* h_urshape = new float2[m_image.getWidth()*m_image.getHeight()];
		float2* h_constraints = new float2[m_image.getWidth()*m_image.getHeight()];
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				h_urshape[i*m_image.getWidth() + j] = make_float2((float)i, (float)j);

				for (unsigned int k = 0; k < constraints.size(); k++)
				{
					if (constraints[k][0] == i && constraints[k][1] == j) h_constraints[i*m_image.getWidth() + j] = make_float2((float)constraints[k][2], (float)constraints[k][3]);
					else												  h_constraints[i*m_image.getWidth() + j] = make_float2(-1, -1);
				}
			}
		}

		cutilSafeCall(cudaMemcpy(d_urshape, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_constraints, h_constraints, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_warpField, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemset(d_warpAngles, 0, sizeof(float3)*m_image.getWidth()*m_image.getHeight()));
		delete h_urshape;
		delete h_constraints;

		m_warpingSolver = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
	}

	~ImageWarping() {
		cutilSafeCall(cudaFree(d_urshape));
		cutilSafeCall(cudaFree(d_warpField));
		cutilSafeCall(cudaFree(d_constraints));
		cutilSafeCall(cudaFree(d_warpAngles));

		SAFE_DELETE(m_warpingSolver);
	}

	ColorImageR32 solve() {
		float weightFit = 0.1f;
		float weightReg = 1.0f;

        unsigned int nonLinearIter = 10;
		unsigned int linearIter = 10;
		m_warpingSolver->solveGN(d_urshape, d_warpField, d_warpAngles, d_constraints, nonLinearIter, linearIter, weightFit, weightReg);

		return copyResultToCPU();
	}

	ColorImageR32 copyResultToCPU() {
		ColorImageR32 result(m_image.getWidth(), m_image.getHeight());

		float2* h_warpField = new float2[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
		
		// apply warp field
		
		delete h_warpField;

		return result;
	}

private:
	ColorImageR32 m_image;

	float2*	d_urshape;
	float2* d_warpField;
	float2* d_constraints;
	float3* d_warpAngles;

	CUDAWarpingSolver*	m_warpingSolver;
};

#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32G32B32A32& image)
	{
		m_image = image;

		cutilSafeCall(cudaMalloc(&d_image,	sizeof(float4)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_target, sizeof(float4)*m_image.getWidth()*m_image.getHeight()));

		float4* h_image  = new float4[m_image.getWidth()*m_image.getHeight()];
		
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				ml::vec4f v = m_image(j, i);
				h_image[i*m_image.getWidth() + j] = make_float4(v.x, v.y, v.z, 255);
			}
		}
		
		cutilSafeCall(cudaMemcpy(d_image, h_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_target, h_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_image;

		m_warpingSolver		 = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
		m_warpingSolverPatch = new CUDAPatchSolverWarping(m_image.getWidth(), m_image.getHeight());
	}

	~ImageWarping()
	{
		cutilSafeCall(cudaFree(d_image));
		cutilSafeCall(cudaFree(d_target));

		SAFE_DELETE(m_warpingSolver);
		SAFE_DELETE(m_warpingSolverPatch);
	}

	ColorImageR32G32B32A32* solve()
	{
		float weightFit = 10.0f;
		float weightReg = 100.0f;
		
		unsigned int nonLinearIter = 1;
		unsigned int linearIter = 50;
		m_warpingSolver->solveGN(d_image, d_target, nonLinearIter, linearIter, weightFit, weightReg);
			
		//unsigned int nonLinearIter = 10;
		//unsigned int patchIter = 16;
		//m_warpingSolverPatch->solveGN(d_image, d_target, nonLinearIter, patchIter, weightFit, weightReg);
		
		copyResultToCPU();

		return &m_result;
	}

	void copyResultToCPU() {
		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		cutilSafeCall(cudaMemcpy(m_result.getPointer(), d_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
	}

private:

	ColorImageR32G32B32A32 m_result;
	ColorImageR32G32B32A32 m_image;
	
	float4*	d_image;
	float4* d_target;
	
	CUDAWarpingSolver*	    m_warpingSolver;
	CUDAPatchSolverWarping* m_warpingSolverPatch;
};

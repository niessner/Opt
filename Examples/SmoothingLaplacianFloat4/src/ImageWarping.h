#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
#include "TerraSolverWarpingFloat4.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32G32B32A32& image)
	{
		m_image = image;

		cutilSafeCall(cudaMalloc(&d_imageFloat4,	sizeof(float4)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_targetFloat4, sizeof(float4)*m_image.getWidth()*m_image.getHeight()));

		cutilSafeCall(cudaMalloc(&d_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_targetFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight()));

		resetGPUMemory();


		m_warpingSolver	= new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
		m_patchSolver = new CUDAPatchSolverWarping(m_image.getWidth(), m_image.getHeight());

		m_terraSolverFloat4 = new TerraSolverWarpingFloat4(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianFloat4AD.t", "gaussNewtonGPU");
		m_terraBlockSolverFloat4 = new TerraSolverWarpingFloat4(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianFloat4AD.t", "gaussNewtonBlockGPU");
		
	}

	void resetGPUMemory()
	{
		float4* h_imageFloat4 = new float4[m_image.getWidth()*m_image.getHeight()];
		float* h_imageFloat = new float[m_image.getWidth()*m_image.getHeight()];

		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				ml::vec4f v = m_image(j, i);
				h_imageFloat4[i*m_image.getWidth() + j] = make_float4(v.x, v.y, v.z, 255);

				float avg = h_imageFloat4[i*m_image.getWidth() + j].x + h_imageFloat4[i*m_image.getWidth() + j].y + h_imageFloat4[i*m_image.getWidth() + j].z;
				h_imageFloat[i*m_image.getWidth() + j] = avg / 3.0f;
			}
		}

		cutilSafeCall(cudaMemcpy(d_imageFloat4, h_imageFloat4, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_targetFloat4, h_imageFloat4, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMemcpy(d_imageFloat, h_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_targetFloat, h_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));

		delete h_imageFloat4;
		delete h_imageFloat;
	}

	~ImageWarping()
	{
		cutilSafeCall(cudaFree(d_imageFloat4));
		cutilSafeCall(cudaFree(d_targetFloat4));

		cutilSafeCall(cudaFree(d_imageFloat));
		cutilSafeCall(cudaFree(d_targetFloat));

		SAFE_DELETE(m_warpingSolver);
		SAFE_DELETE(m_patchSolver);
		SAFE_DELETE(m_terraSolverFloat4);
		SAFE_DELETE(m_terraBlockSolverFloat4);
	}

	ColorImageR32G32B32A32* solve()
	{
		float weightFit = 0.1f;
		float weightReg = 1.0f;
		
		unsigned int nonLinearIter = 10;
		unsigned int linearIter = 10;
		unsigned int patchIter = 10;

		std::cout << "CUDA" << std::endl;
		resetGPUMemory();
		m_warpingSolver->solveGN(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();
		
		std::cout << "\n\nCUDA_PATCH" << std::endl;
		resetGPUMemory();
		m_patchSolver->solveGN(d_imageFloat4, d_targetFloat4, nonLinearIter, patchIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();

		std::cout << "\n\nTERRA_FLOAT4" << std::endl;
		resetGPUMemory();
		m_terraSolverFloat4->solve(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();

		std::cout << "\n\nTERRA_BLOCK" << std::endl;
		resetGPUMemory();
		m_terraBlockSolverFloat4->solve(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat4();		

		return &m_result;
	}

	

	void copyResultToCPUFromFloat4() {
		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		cutilSafeCall(cudaMemcpy(m_result.getPointer(), d_imageFloat4, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
	}

	void copyResultToCPUFromFloat() {
		float* h_result = new float[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_result, d_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
		for (unsigned int i = 0; i < m_image.getWidth()*m_image.getHeight(); i++) {
			float v = h_result[i];
			m_result.getPointer()[i] = vec4f(v,v,v, 1.0f);
		}

		delete h_result;
	}

private:

	ColorImageR32G32B32A32 m_result;
	ColorImageR32G32B32A32 m_image;
	
	float4*	d_imageFloat4;
	float4* d_targetFloat4;
	
	CUDAWarpingSolver*			m_warpingSolver;
	CUDAPatchSolverWarping*		m_patchSolver;
	TerraSolverWarpingFloat4*	m_terraSolverFloat4; 
	TerraSolverWarpingFloat4*	m_terraBlockSolverFloat4;


	float* d_imageFloat;
	float* d_targetFloat;
};

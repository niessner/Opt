#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
//#include "TerraSolverWarpingFloat4.h"
#include "CuspSparseLaplacianSolverLinearOp.h"

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
		m_terraSolver = new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianAD.t", "gradientDescentGPU");
		m_terraBlockSolver = new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacianAD.t", "gaussNewtonBlockGPU");
		//m_terraSolverFloat4 = new TerraSolverWarpingFloat4(m_image.getWidth(), m_image.getHeight(), "smoothingLaplacian4AD.t", "gaussNewtonGPU");
		m_cuspSolverFloat = new CuspSparseLaplacianSolverLinearOp(m_image.getWidth(), m_image.getHeight());
	}

	void resetGPUMemory()
	{
		float4* h_image = new float4[m_image.getWidth()*m_image.getHeight()];
		float* h_imageFloat = new float[m_image.getWidth()*m_image.getHeight()];

		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				ml::vec4f v = m_image(j, i);
				h_image[i*m_image.getWidth() + j] = make_float4(v.x, v.y, v.z, 255);

				float avg = h_image[i*m_image.getWidth() + j].x + h_image[i*m_image.getWidth() + j].y + h_image[i*m_image.getWidth() + j].z;
				h_imageFloat[i*m_image.getWidth() + j] = avg / 3.0f;
			}
		}

		cutilSafeCall(cudaMemcpy(d_imageFloat4, h_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_targetFloat4, h_image, sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMemcpy(d_imageFloat, h_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_targetFloat, h_imageFloat, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));

		delete h_image;
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
		SAFE_DELETE(m_terraSolver);
		SAFE_DELETE(m_terraBlockSolver);

//		SAFE_DELETE(m_terraSolverFloat4);

		SAFE_DELETE(m_cuspSolverFloat);
	}

	ColorImageR32G32B32A32* solve()
	{
		float weightFit = 0.1f;
		float weightReg = 100.0f;
		
		//unsigned int nonLinearIter = 10;
		//unsigned int linearIter = 10;
		//m_warpingSolver->solveGN(d_image, d_target, nonLinearIter, linearIter, weightFit, weightReg);
			
		//unsigned int nonLinearIter = 10;
		//unsigned int patchIter = 16;
		//m_warpingSolverPatch->solveGN(d_image, d_target, nonLinearIter, patchIter, weightFit, weightReg);
		
		//copyResultToCPU();


		unsigned int nonLinearIter = 10;
		unsigned int linearIter = 10;
		unsigned int patchIter = 16;

		
		//std::cout << "CUDA" << std::endl;
		//m_warpingSolver->solveGN(d_imageFloat, d_targetFloat, nonLinearIter, linearIter, weightFit, weightReg);
		//copyResultToCPUFromFloat();

		//std::cout << "CUDA_PATCH" << std::endl;
		//resetGPUMemory();
		//m_patchSolver->solveGN(d_imageFloat, d_targetFloat, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		//copyResultToCPUFromFloat();

		//std::cout << "\n\nTERRA" << std::endl;
		//resetGPUMemory();
		//m_terraSolver->solve(d_imageFloat, d_targetFloat, nonLinearIter, linearIter, weightFit, weightReg);
		//copyResultToCPUFromFloat();

		//std::cout << "\n\nTERRA_BLOCK" << std::endl;
		//resetGPUMemory();
		//m_terraBlockSolver->solve(d_imageFloat, d_targetFloat, nonLinearIter, linearIter, weightFit, weightReg);
		//copyResultToCPUFromFloat();
		
		std::cout << "CUSP" << std::endl;
		m_cuspSolverFloat->solvePCG(d_imageFloat, d_targetFloat, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat();

		//resetGPUMemory();
		//m_terraSolverFloat4->solve(d_imageFloat4, d_targetFloat4, nonLinearIter, linearIter, weightFit, weightReg);
		//copyResultToCPUFromFloat4();

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
	TerraSolverWarping*			m_terraSolver; 
	TerraSolverWarping*			m_terraBlockSolver;
//	TerraSolverWarpingFloat4*	m_terraSolverFloat4; 
	CuspSparseLaplacianSolverLinearOp* m_cuspSolverFloat;


	float* d_imageFloat;
	float* d_targetFloat;
};

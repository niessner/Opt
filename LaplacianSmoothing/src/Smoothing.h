
#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDALaplacianSolver.h"
#include "CUSparseLaplacianSolver.h"

class Smoothing {
public:
	Smoothing(const ColorImageR32& image) {
		m_image = image;
		cutilSafeCall(cudaMalloc(&d_image, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_result, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMemcpy(d_image, m_image.getPointer(), sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		
		float* h_result = new float[m_image.getWidth()*m_image.getHeight()];
		memset(h_result, 0, sizeof(float)*m_image.getWidth()*m_image.getHeight());
		cutilSafeCall(cudaMemcpy(d_result, h_result, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		SAFE_DELETE_ARRAY(h_result);

		m_laplacianSolver = new CUDALaplacianSolver(m_image.getWidth(), m_image.getHeight());
		m_cusparseLaplacianSolver = new CUSparseLaplacianSolver(m_image.getWidth(), m_image.getHeight());
	}

	~Smoothing() {
		cutilSafeCall(cudaFree(d_image));
		cutilSafeCall(cudaFree(d_result));
		SAFE_DELETE(m_laplacianSolver);
		SAFE_DELETE(m_cusparseLaplacianSolver);
	}

	ColorImageR32 solve() {
		float weightFit = 0.1f;
		float weightReg = 1.0f;

        unsigned int nonLinearIter = 32;
		unsigned int linearIter = 1000;
		m_laplacianSolver->solveGN(d_image, d_result, nonLinearIter, linearIter, weightFit, weightReg);
		//m_laplacianSolver->solveGD(d_image, d_result, nonLinearIter, weightFit, weightReg);
		//m_cusparseLaplacianSolver->solvePCG(d_image, d_result, linearIter, weightFit, weightReg);
		return copyResultToCPU();
	}

	ColorImageR32 copyResultToCPU() {
		ColorImageR32 result(m_image.getWidth(), m_image.getHeight());
		cutilSafeCall(cudaMemcpy(result.getPointer(), d_result, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
		return result;
	}

private:
	ColorImageR32 m_image;
	float* d_image;
	float* d_result;

	CUDALaplacianSolver*	 m_laplacianSolver;
	CUSparseLaplacianSolver* m_cusparseLaplacianSolver;
};

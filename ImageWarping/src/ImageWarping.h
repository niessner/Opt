
#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
class ImageWarping {
public:
	ImageWarping(const ColorImageR32& image, std::vector<std::vector<int>>& constraints) {
		m_image = image;
		cutilSafeCall(cudaMalloc(&d_image, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_result, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMemcpy(d_image, m_image.getPointer(), sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		
		float* h_result = new float[m_image.getWidth()*m_image.getHeight()];
		memset(h_result, 0, sizeof(float)*m_image.getWidth()*m_image.getHeight());

		cutilSafeCall(cudaMemcpy(d_result, m_image.getPointer(), sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		SAFE_DELETE_ARRAY(h_result);

		m_warpingSolver = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
	}

	~ImageWarping() {
		cutilSafeCall(cudaFree(d_image));
		cutilSafeCall(cudaFree(d_result));
		SAFE_DELETE(m_warpingSolver);
	}

	ColorImageR32 solve() {
		float weightFit = 0.1f;
		float weightReg = 1.0f;

        unsigned int nonLinearIter = 1;
		unsigned int linearIter = 100;
		m_warpingSolver->solveGN(d_image, d_result, nonLinearIter, linearIter, weightFit, weightReg);

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

	CUDAWarpingSolver*	m_warpingSolver;
};

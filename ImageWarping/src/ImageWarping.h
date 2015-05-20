#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32& image, const ColorImageR32& imageMask, std::vector<std::vector<int>>& constraints) : m_constraints(constraints){
		m_image = image;
		m_imageMask = imageMask;

		cutilSafeCall(cudaMalloc(&d_urshape,    sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpField,  sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpAngles, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_constraints, sizeof(float2)*m_image.getWidth()*m_image.getHeight()));

		float2* h_urshape = new float2[m_image.getWidth()*m_image.getHeight()];
		
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				h_urshape[i*m_image.getWidth() + j] = make_float2((float)i, (float)j);
			}
		}

		setConstraintImage(0.5f);

		cutilSafeCall(cudaMemcpy(d_urshape, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_warpField, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemset(d_warpAngles, 0, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		delete h_urshape;

		m_warpingSolver = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());
	}

	void setConstraintImage(float alpha)
	{
		float2* h_constraints = new float2[m_image.getWidth()*m_image.getHeight()];
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				h_constraints[i*m_image.getWidth() + j] = make_float2(-1, -1);
				for (unsigned int k = 0; k < m_constraints.size(); k++)
				{
					if (m_constraints[k][0] == i && m_constraints[k][1] == j)
					{
						if (m_imageMask(i, j) == 0)
						{
							float y = (1.0f - alpha)*(float)i + alpha*(float)m_constraints[k][2];
							float x = (1.0f - alpha)*(float)j + alpha*(float)m_constraints[k][3];


							h_constraints[i*m_image.getWidth() + j] =  make_float2(y, x);
						}
					}
				}
			}
		}

		cutilSafeCall(cudaMemcpy(d_constraints, h_constraints, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_constraints;
	}

	~ImageWarping() {
		cutilSafeCall(cudaFree(d_urshape));
		cutilSafeCall(cudaFree(d_warpField));
		cutilSafeCall(cudaFree(d_constraints));
		cutilSafeCall(cudaFree(d_warpAngles));

		SAFE_DELETE(m_warpingSolver);
	}

	ColorImageR32 solve() {
		float weightFit = 100.0f;
		float weightReg = 0.01f;

		for (unsigned int i = 0; i < 10; i++)
		{
			setConstraintImage((float)i/(float)9);

			unsigned int nonLinearIter = 10;
			unsigned int linearIter = 50;
			m_warpingSolver->solveGN(d_urshape, d_warpField, d_warpAngles, d_constraints, nonLinearIter, linearIter, weightFit, weightReg);
		}

		return copyResultToCPU();
	}

	ColorImageR32 copyResultToCPU() {
		ColorImageR32 result(m_image.getWidth(), m_image.getHeight());

		float2* h_warpField = new float2[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
		
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				float2 pos = h_warpField[i*m_image.getWidth()+j];

				unsigned int x = (unsigned int)(pos.x + 0.5f);
				unsigned int y = (unsigned int)(pos.y + 0.5f);

				if (x < m_image.getHeight() && y < m_image.getWidth()) result(x, y) = m_image(i, j);
				else												   result(x, y) = 0;
			}
		}
		
		delete h_warpField;

		return result;
	}

private:
	ColorImageR32 m_image;
	ColorImageR32 m_imageMask;

	float2*	d_urshape;
	float2* d_warpField;
	float2* d_constraints;
	float*  d_warpAngles;

	std::vector<std::vector<int>>& m_constraints;

	CUDAWarpingSolver*	m_warpingSolver;
};

#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
#include "CeresSolverImageWarping.h"


static bool useCUDA = true;
static bool useTerra = true;
static bool useAD = true;
static bool useCeres = true;

class ImageWarping {
public:
	ImageWarping(const ColorImageR32& image, const ColorImageR32& imageMask, std::vector<std::vector<int>>& constraints) : m_constraints(constraints){
		m_image = image;
		m_imageMask = imageMask;

		cutilSafeCall(cudaMalloc(&d_urshape,     sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpField,   sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_warpAngles,  sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_constraints, sizeof(float2)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMalloc(&d_mask,		 sizeof(float)*m_image.getWidth()*m_image.getHeight()));

		resetGPU();

		if (useCUDA)
		  m_warpingSolver		 = new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight());

		if (useTerra)
		  m_warpingSolverTerra = new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarping.t", "gaussNewtonGPU");

		if (useAD)
		  m_warpingSolverTerraAD = new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarpingAD.t", "gaussNewtonGPU");

        if (useCeres)
            m_warpingSolverCeres = new CeresSolverWarping(m_image.getWidth(), m_image.getHeight());

	}

	void resetGPU()
	{
		float2* h_urshape = new float2[m_image.getWidth()*m_image.getHeight()];
		float*  h_mask = new float[m_image.getWidth()*m_image.getHeight()];

		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				h_urshape[i*m_image.getWidth() + j] = make_float2((float)i, (float)j);
				h_mask[i*m_image.getWidth() + j] = m_imageMask(i, j);
			}
		}

		setConstraintImage(1.0f);

		cutilSafeCall(cudaMemcpy(d_urshape, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_warpField, h_urshape, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemset(d_warpAngles, 0, sizeof(float)*m_image.getWidth()*m_image.getHeight()));
		cutilSafeCall(cudaMemcpy(d_mask, h_mask, sizeof(float)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_urshape;
		delete h_mask;
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


		if (useCUDA) 
		  SAFE_DELETE(m_warpingSolver);
		
		if (useTerra) 
		  SAFE_DELETE(m_warpingSolverTerra);
		
		if (useAD) 
		  SAFE_DELETE(m_warpingSolverTerraAD);
		

	}

	ColorImageR32* solve() {
		float weightFit = 100.0f;
		float weightReg = 0.01f;

		unsigned int numIter = 2;
		unsigned int nonLinearIter = 3;
		unsigned int linearIter = 3;
		unsigned int patchIter = 32;

		//unsigned int numIter = 20;
		//unsigned int nonLinearIter = 32;
		//unsigned int linearIter = 50;
		//unsigned int patchIter = 32;

		if (useCUDA) {
		  resetGPU();
		  for (unsigned int i = 0; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION"  << i << "  (CUDA) ///////////////" << std::endl;
		    setConstraintImage((float)i / (float)20);

		    m_warpingSolver->solveGN(d_urshape, d_warpField, d_warpAngles, d_constraints, d_mask, nonLinearIter, linearIter, weightFit, weightReg);
		    
			//	std::cout << std::endl;
		  }
		  copyResultToCPU();
		}

		if (useTerra) {
		  resetGPU();

		  std::cout << std::endl << std::endl;
		  
		  for (unsigned int i = 0; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION"  << i << "  (TERRA) ///////////////" << std::endl;
		    setConstraintImage((float)i / (float)20);
		    
		    m_warpingSolverTerra->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);

			//	std::cout << std::endl;
		  }
		  copyResultToCPU();
		}


		if (useAD) {
		  resetGPU();

		  std::cout << std::endl << std::endl;
		
		  for (unsigned int i = 0; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION" << i << "  (DSL AD) ///////////////" << std::endl;
		    setConstraintImage((float)i / (float)20);

			
		    m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		    std::cout << std::endl;
		  }

		  copyResultToCPU();
		}

        if (useCeres) {
            resetGPU();

            const int pixelCount = m_image.getWidth()*m_image.getHeight();
            
            float2* h_warpField = new float2[pixelCount];
            float* h_warpAngles = new float[pixelCount];

            float2* h_urshape = new float2[pixelCount];
            float*  h_mask = new float[pixelCount];
            float2* h_constraints = new float2[pixelCount];

            cutilSafeCall(cudaMemcpy(h_urshape, d_urshape, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_mask, d_mask, sizeof(float) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));

            std::cout << std::endl << std::endl;

            for (unsigned int i = 0; i < numIter; i++)	{
                std::cout << "//////////// ITERATION" << i << "  (CERES) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)20);
                cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));

                //m_warpingSolverCeres->solve(h_warpField, h_warpAngles, h_urshape, h_constraints, h_mask, weightFit, weightReg);
                std::cout << std::endl;
            }
        }

		return &m_result;
	}

	void copyResultToCPU() {
		m_result = ColorImageR32(m_image.getWidth(), m_image.getHeight());
		m_result.setPixels(255);

		float2* h_warpField = new float2[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

		unsigned int c = 3;
		for (unsigned int i = 0; i < m_image.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image.getWidth(); j++)
			{
				if (i + 1 < m_image.getHeight() && j + 1 < m_image.getWidth())
				{
					if (m_imageMask(i, j) == 0)
					{
						float2 pos00 = h_warpField[i*m_image.getWidth() + j];
						float2 pos01 = h_warpField[i*m_image.getWidth() + (j + 1)];
						float2 pos10 = h_warpField[(i + 1)*m_image.getWidth() + j];
						float2 pos11 = h_warpField[(i + 1)*m_image.getWidth() + (j + 1)];

						float v00 = m_image(i, j);
						float v01 = m_image(i, (j + 1));
						float v10 = m_image((i + 1), j);
						float v11 = m_image((i + 1), (j + 1));

						for (unsigned int g = 0; g < c; g++)
						{
							for (unsigned int h = 0; h < c; h++)
							{
								float alpha = (float)g / (float)c;
								float beta = (float)h / (float)c;

								bool valid00 = (m_imageMask(i, j) == 0);
								bool valid01 = (m_imageMask(i, j+1) == 0);
								bool valid10 = (m_imageMask(i+1, j) == 0);
								bool valid11 = (m_imageMask(i+1, j+1) == 0);

								if (valid00 && valid01 && valid10 && valid11)
								{
									float2 pos0 = (1 - alpha)*pos00 + alpha* pos01;
									float2 pos1 = (1 - alpha)*pos10 + alpha* pos11;
									float2 pos = (1 - beta)*pos0 + beta*pos1;

									float v0 = (1 - alpha)*v00 + alpha* v01;
									float v1 = (1 - alpha)*v10 + alpha* v11;
									float v = (1 - beta)*v0 + beta * v1;

									unsigned int x = (unsigned int)(pos.x + 0.5f);
									unsigned int y = (unsigned int)(pos.y + 0.5f);
									if (x < m_result.getHeight() && y < m_result.getWidth()) m_result(x, y) = v;
								}
								else
								{
									float2 pos = pos00;
									float v = v00;
									unsigned int x = (unsigned int)(pos.x + 0.5f);
									unsigned int y = (unsigned int)(pos.y + 0.5f);
									if (x < m_result.getHeight() && y < m_result.getWidth()) m_result(x, y) = v;
								}
							}
						}
					}
				}
			}
		}
		
		delete h_warpField;
	}

private:
	ColorImageR32 m_image;
	ColorImageR32 m_imageMask;

	ColorImageR32 m_result;

	float2*	d_urshape;
	float2* d_warpField;
	float2* d_constraints;
	float*  d_warpAngles;
	float*  d_mask;

	std::vector<std::vector<int>>& m_constraints;

	CUDAWarpingSolver*	    m_warpingSolver;

	TerraSolverWarping*		m_warpingSolverTerraAD;

	TerraSolverWarping*		m_warpingSolverTerra;

    CeresSolverWarping*		m_warpingSolverCeres;
};

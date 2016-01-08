#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
#include "CeresSolverImageWarping.h"


static bool useCUDA = false;
static bool useTerra = false;
static bool useAD = true;
static bool useCeres = false;

class ImageWarping {
public:
    ImageWarping(const ColorImageR32& image, const ColorImageR32G32B32& imageColor, const ColorImageR32& imageMask, std::vector<std::vector<int>>& constraints) : m_constraints(constraints){
		m_image = image;
        m_imageColor = imageColor;
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

		for (unsigned int y = 0; y < m_image.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_image.getWidth(); x++)
			{
				h_urshape[y*m_image.getWidth() + x] = make_float2((float)x, (float)y);
				h_mask[y*m_image.getWidth() + x] = m_imageMask(x, y);
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
        //printf("m_constraints.size() = %d\n", m_constraints.size());
        for (unsigned int y = 0; y < m_image.getHeight(); y++)
        {
            for (unsigned int x = 0; x < m_image.getWidth(); x++)
            {
				h_constraints[y*m_image.getWidth() + x] = make_float2(-1, -1);
			}
		}

        for (unsigned int k = 0; k < m_constraints.size(); k++)
        {
            int x = m_constraints[k][0];
            int y = m_constraints[k][1];
            
            if (m_imageMask(x, y) == 0)
            {
                float newX = (1.0f - alpha)*(float)x + alpha*(float)m_constraints[k][2];
                float newY = (1.0f - alpha)*(float)y + alpha*(float)m_constraints[k][3];


                h_constraints[y*m_image.getWidth() + x] = make_float2(newX, newY);
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

	ColorImageR32G32B32* solve() {
		float weightFit = 100.0f;
		float weightReg = 0.01f;

		unsigned int numIter = 20;
		unsigned int nonLinearIter = 2;
		unsigned int linearIter = 200;
		unsigned int patchIter = 32;

		//unsigned int numIter = 20;
		//unsigned int nonLinearIter = 32;
		//unsigned int linearIter = 50;
		//unsigned int patchIter = 32;

		if (useCUDA) {
		  resetGPU();
		  for (unsigned int i = 0; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION"  << i << "  (CUDA) ///////////////" << std::endl;
            setConstraintImage((float)i / (float)numIter);

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
            setConstraintImage((float)i / (float)numIter);
		    
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
            setConstraintImage((float)i / (float)numIter);

			
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

            float totalCeresTimeMS = 0.0f;

            cutilSafeCall(cudaMemcpy(h_urshape, d_urshape, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_mask, d_mask, sizeof(float) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_warpAngles, d_warpAngles, sizeof(float) * pixelCount, cudaMemcpyDeviceToHost));

            std::cout << std::endl << std::endl;

            for (unsigned int i = 0; i < numIter; i++)	{
                std::cout << "//////////// ITERATION" << i << "  (CERES) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)numIter);
                cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(float2) * pixelCount, cudaMemcpyDeviceToHost));

                totalCeresTimeMS = m_warpingSolverCeres->solve(h_warpField, h_warpAngles, h_urshape, h_constraints, h_mask, weightFit, weightReg);
                std::cout << std::endl;
            }

            cutilSafeCall(cudaMemcpy(d_warpField, h_warpField, sizeof(float2) * pixelCount, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_warpAngles, h_warpAngles, sizeof(float) * pixelCount, cudaMemcpyHostToDevice));
            copyResultToCPU();

            std::cout << "Ceres time for final iteration: " << totalCeresTimeMS << "ms" << std::endl;

            std::cout << "testing CERES cost function by calling AD..." << std::endl;

            //m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
            std::cout << std::endl;
        }

        return &m_resultColor;
	}

	void copyResultToCPU() {
        m_resultColor = ColorImageR32G32B32(m_image.getWidth(), m_image.getHeight());
        m_resultColor.setPixels(vec3f(255.0f, 255.0f, 255.0f));

		float2* h_warpField = new float2[m_image.getWidth()*m_image.getHeight()];
		cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(float2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

		unsigned int c = 3;
		for (unsigned int y = 0; y < m_image.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_image.getWidth(); x++)
			{
				if (y + 1 < m_image.getHeight() && x + 1 < m_image.getWidth())
				{
					if (m_imageMask(x, y) == 0)
					{
						float2 pos00 = h_warpField[y*m_image.getWidth() + x];
						float2 pos01 = h_warpField[y*m_image.getWidth() + (x + 1)];
						float2 pos10 = h_warpField[(y + 1)*m_image.getWidth() + x];
						float2 pos11 = h_warpField[(y + 1)*m_image.getWidth() + (x + 1)];

                        vec3f v00 = m_imageColor(x, y);
                        vec3f v01 = m_imageColor(x, (y + 1));
                        vec3f v10 = m_imageColor((x + 1), y);
                        vec3f v11 = m_imageColor((x + 1), (y + 1));

						for (unsigned int g = 0; g < c; g++)
						{
							for (unsigned int h = 0; h < c; h++)
							{
								float alpha = (float)g / (float)c;
								float beta = (float)h / (float)c;

								bool valid00 = (m_imageMask(x,   y) == 0);
								bool valid01 = (m_imageMask(x,   y+1) == 0);
								bool valid10 = (m_imageMask(x+1, y) == 0);
								bool valid11 = (m_imageMask(x+1, y+1) == 0);

								if (valid00 && valid01 && valid10 && valid11)
								{
									float2 pos0 = (1 - alpha)*pos00 + alpha* pos01;
									float2 pos1 = (1 - alpha)*pos10 + alpha* pos11;
									float2 pos = (1 - beta)*pos0 + beta*pos1;

									vec3f v0 = (1 - alpha)*v00 + alpha* v01;
                                    vec3f v1 = (1 - alpha)*v10 + alpha* v11;
                                    vec3f v = (1 - beta)*v0 + beta * v1;

									unsigned int newX = (unsigned int)(pos.x + 0.5f);
                                    unsigned int newY = (unsigned int)(pos.y + 0.5f);
                                    if (newX < m_resultColor.getWidth() && newY < m_resultColor.getHeight()) m_resultColor(newX, newY) = v;
								}
								else
								{
									float2 pos = pos00;
									vec3f v = v00;
                                    unsigned int newX = (unsigned int)(pos.x + 0.5f);
                                    unsigned int newY = (unsigned int)(pos.y + 0.5f);
                                    if (newX < m_resultColor.getWidth() && newY < m_resultColor.getHeight()) m_resultColor(newX, newY) = v;
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
    ColorImageR32G32B32 m_imageColor;
	ColorImageR32 m_imageMask;

	ColorImageR32 m_result;
    ColorImageR32G32B32 m_resultColor;

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

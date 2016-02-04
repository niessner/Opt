#pragma once

#include "mLibInclude.h"
//#include "/Users/zdevito/vdb/vdb.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
#include "CeresSolverImageWarping.h"


static bool useCUDA = true;
static bool useTerra = true;
static bool useAD = true;
static bool useCeres = false;

static bool earlyOut = true;


static bool
PointInTriangleBarycentric(float x0, float y0, float w0,
                           float x1, float y1, float w1,
                           float x2, float y2, float w2,
                           float sx, float sy, 
                           float *wt0, float *wt1, float *wt2) {
    x0 /= w0;
    y0 /= w0;
    x1 /= w1;
    y1 /= w1;
    x2 /= w2;
    y2 /= w2;

    float v0x = x2 - x0, v0y = y2 - y0;
    float v1x = x1 - x0, v1y = y1 - y0;
    float v2x = sx - x0, v2y = sy - y0;

    float area = 0.5f * (v1x * v0y - v1y * v0x);
    if (area <= 0.) {
        // backfacing
        return false;
    }

#define DOT2(a,b) ((a##x)*(b##x)+(a##y)*(b##y))
    float dot00 = DOT2(v0, v0);
    float dot01 = DOT2(v0, v1);
    float dot11 = DOT2(v1, v1);
    float denom = (dot00 * dot11 - dot01 * dot01);
    if (denom == 0)
        return false;
    float invDenom = 1.f / denom;

    float dot02 = DOT2(v0, v2);
    float dot12 = DOT2(v1, v2);

    // Compute barycentric coordinates
    float b2 = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float b1 = (dot00 * dot12 - dot01 * dot02) * invDenom;
    float b0 = 1.f - b1 - b2;

    *wt0 = b0;
    *wt1 = b1;
    *wt2 = b2;

    return (b0 > 0. && b1 > 0 && b2 > 0);
}

inline bool
PointInTriangleLK(float x0, float y0, float w0,
                  float x1, float y1, float w1,
                  float x2, float y2, float w2,
                  float sx, float sy, 
                  float *wt0, float *wt1, float *wt2) {
    float X[3], Y[3];

    X[0] = x0 - sx*w0;
    X[1] = x1 - sx*w1;
    X[2] = x2 - sx*w2;

    Y[0] = y0 - sy*w0;
    Y[1] = y1 - sy*w1;
    Y[2] = y2 - sy*w2;

    float d01 = X[0]*Y[1] - Y[0]*X[1];
    float d12 = X[1]*Y[2] - Y[1]*X[2];
    float d20 = X[2]*Y[0] - Y[2]*X[0];

    if (d01 < 0 & d12 < 0 & d20 < 0) {
        //printf("Backfacing\n");
        // backfacing
        return false;
    }

    float OneOverD = 1.f / (d01 + d12 + d20);
    d01 *= OneOverD;
    d12 *= OneOverD;
    d20 *= OneOverD;

    *wt0 = d12;
    *wt1 = d20;
    *wt2 = d01;

    return (d01 >= 0 && d12 >= 0 && d20 >= 0);
}

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
		unsigned int nonLinearIter = 3;
		unsigned int linearIter = 200;
		unsigned int patchIter = 32;

		//unsigned int numIter = 20;
		//unsigned int nonLinearIter = 32;
		//unsigned int linearIter = 50;
		//unsigned int patchIter = 32;

		if (useCUDA) {
		  resetGPU();
		  for (unsigned int i = 1; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION"  << i << "  (CUDA) ///////////////" << std::endl;
            setConstraintImage((float)i / (float)numIter);

		    m_warpingSolver->solveGN(d_urshape, d_warpField, d_warpAngles, d_constraints, d_mask, nonLinearIter, linearIter, weightFit, weightReg);
		    if (earlyOut) break;
			//	std::cout << std::endl;
		  }
		  copyResultToCPU();
		}

		if (useTerra) {
		  resetGPU();

		  std::cout << std::endl << std::endl;
		  
		  for (unsigned int i = 1; i < numIter; i++)	{
		    std::cout << "//////////// ITERATION"  << i << "  (TERRA) ///////////////" << std::endl;
            setConstraintImage((float)i / (float)numIter);
		    
		    m_warpingSolverTerra->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		    if (earlyOut) break;
			//	std::cout << std::endl;
		  }
		  copyResultToCPU();
		}


		if (useAD) {
		  resetGPU();

		  std::cout << std::endl << std::endl;
		
		  for (unsigned int i = 1; i < numIter; i++)	{
		    
		    std::cout << "//////////// ITERATION" << i << "  (DSL AD) ///////////////" << std::endl;
            setConstraintImage((float)i / (float)numIter);
			
		    m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
		    std::cout << std::endl;
            if (earlyOut) break;
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

    vec2f toVec2(float2 p) {
        return vec2f(p.x, p.y);
    }

    void rasterizeTriangle(float2 p0, float2 p1, float2 p2, vec3f c0, vec3f c1, vec3f c2) {
        vec2f t0 = toVec2(p0)*m_scale;
        vec2f t1 = toVec2(p1)*m_scale;
        vec2f t2 = toVec2(p2)*m_scale;


        int W = m_resultColor.getWidth();
        int H = m_resultColor.getHeight();

        vec2f minBound = math::floor(math::min(t0, math::min(t1, t2)));
        vec2f maxBound = math::ceil(math::max(t0, math::max(t1, t2)));
        for (int x = (int)minBound.x; x <= maxBound.x; ++x) {
            for (int y = (int)minBound.y; y <= maxBound.y; ++y) {
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    float b0, b1, b2;
                    if (PointInTriangleLK(t0.x, t0.y, 1.0f,
                        t1.x, t1.y, 1.0f,
                        t2.x, t2.y, 1.0f, x, y, &b0, &b1, &b2)) {
                        vec3f color = c0*b0 + c1*b1 + c2*b2;
                        m_resultColor(x, y) = color;
                    }

                }
            }
        }

        //bound
        //loop
        //point in trinagle
        // z-test?
    }

	void copyResultToCPU() {
        m_scale = 1.0f;
        m_resultColor = ColorImageR32G32B32(m_image.getWidth()*m_scale, m_image.getHeight()*m_scale);
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
                        vec3f v01 = m_imageColor(x+1, y);
                        vec3f v10 = m_imageColor(x, y+1);
                        vec3f v11 = m_imageColor(x + 1, y + 1);
                        
             			bool valid00 = (m_imageMask(x,   y) == 0);
						bool valid01 = (m_imageMask(x,   y+1) == 0);
						bool valid10 = (m_imageMask(x+1, y) == 0);
						bool valid11 = (m_imageMask(x+1, y+1) == 0);
                        
                        if (valid00 && valid01 && valid10 && valid11) {
                            rasterizeTriangle(pos00, pos01, pos10, 
                                v00, v01, v10);
                            rasterizeTriangle(pos10, pos01, pos11, 
                                v10, v01, v11);
                        }
                        /*
						for (unsigned int g = 0; g < c; g++)
						{
							for (unsigned int h = 0; h < c; h++)
							{
								float alpha = (float)g / (float)c;
								float beta = (float)h / (float)c;



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
						}*/
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

    float m_scale;

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

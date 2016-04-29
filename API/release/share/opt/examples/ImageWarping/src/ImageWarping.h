#pragma once

#include "mLibInclude.h"

#include "cudaUtil.h"
#include <cuda_runtime.h>

#include "TerraSolverWarping.h"


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

    if ((d01 < 0) & (d12 < 0) & (d20 < 0)) {
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

		
		m_warpingSolverTerraAD = new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarpingAD.t", "gaussNewtonGPU");

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


		SAFE_DELETE(m_warpingSolverTerraAD);
		
	}

	ColorImageR32G32B32* solve() {
		float weightFit = 100.0f;
		float weightReg = 0.01f;

		unsigned int numIter = 10;
		unsigned int nonLinearIter = 3;
		unsigned int linearIter = 200;

        resetGPU();

        std::cout << std::endl << std::endl;

        for (unsigned int i = 1; i < numIter; i++)	{

            std::cout << "//////////// ITERATION" << i << "  (DSL AD) ///////////////" << std::endl;
            setConstraintImage((float)i / (float)numIter);

            m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, weightFit, weightReg);
            std::cout << std::endl;
        }

        copyResultToCPU();
		
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
                        t2.x, t2.y, 1.0f, (float)x, (float)y, &b0, &b1, &b2)) {
                        vec3f color = c0*b0 + c1*b1 + c2*b2;
                        m_resultColor(x, y) = color;
                    }

                }
            }
        }
    }

	void copyResultToCPU() {
        m_scale = 1.0f;
        m_resultColor = ColorImageR32G32B32(m_image.getWidth()*(int)m_scale, m_image.getHeight()*(int)m_scale);
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

	TerraSolverWarping*		m_warpingSolverTerraAD;
};
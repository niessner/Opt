#pragma once
#include "../../shared/SolverIteration.h"
#include "mLibInclude.h"
//#include "/Users/zdevito/vdb/vdb.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "TerraSolverWarping.h"
#include "CeresSolverImageWarping.h"
#include "../../shared/CombinedSolverParameters.h"
#include "Configure.h"
#include "../../shared/Precision.h"
#include <fstream>



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

	float d01 = X[0] * Y[1] - Y[0] * X[1];
	float d12 = X[1] * Y[2] - Y[1] * X[2];
	float d20 = X[2] * Y[0] - Y[2] * X[0];

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
	ImageWarping(const ColorImageR32& image, const ColorImageR32G32B32& imageColor, const ColorImageR32& imageMask, std::vector<std::vector<int>>& constraints, bool performanceRun, bool lmOnlyFullSolve) : m_constraints(constraints){
		m_image = image;
		m_imageColor = imageColor;
		m_imageMask = imageMask;

		cutilSafeCall(cudaMalloc(&d_urshape, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight()));
        cutilSafeCall(cudaMalloc(&d_warpField, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight()));
        cutilSafeCall(cudaMalloc(&d_warpAngles, sizeof(OPT_FLOAT)*m_image.getWidth()*m_image.getHeight()));
        cutilSafeCall(cudaMalloc(&d_constraints, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight()));
        cutilSafeCall(cudaMalloc(&d_mask, sizeof(OPT_FLOAT)*m_image.getWidth()*m_image.getHeight()));

		resetGPU();


		m_params.numIter = 20;
		m_params.useCUDA = false;
		m_params.nonLinearIter = 8;
		m_params.linearIter = 400;
		if (performanceRun) {
			m_params.useCUDA = false;
			m_params.useTerra = false;
			m_params.useOpt = true;
            m_params.useOptLM = true;
			m_params.useCeres = true;
			m_params.earlyOut = true;
		}
		if (lmOnlyFullSolve) {
			m_params.useCUDA = false;
            m_params.useOpt = false;
            m_params.useOptLM = true;
			m_params.linearIter = 500;// m_image.getWidth()*m_image.getHeight();
			if (image.getWidth() > 1024) {
				m_params.nonLinearIter = 100;
			}
// TODO: Remove for < 2048x2048
#if !USE_CERES_PCG
			//m_params.useCeres = false;
#endif
		}
		m_lmOnlyFullSolve = lmOnlyFullSolve;

        if (m_params.useCUDA)
            m_warpingSolver = std::unique_ptr<CUDAWarpingSolver>(new CUDAWarpingSolver(m_image.getWidth(), m_image.getHeight()));

        if (m_params.useTerra)
            m_warpingSolverTerra = std::unique_ptr<TerraSolverWarping>(new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarping.t", "gaussNewtonGPU"));

        if (m_params.useOpt) {
            m_warpingSolverTerraAD = std::unique_ptr<TerraSolverWarping>(new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarpingAD.t", "gaussNewtonGPU"));
		}

        if (m_params.useOptLM) {
            m_warpingSolverTerraLMAD = std::unique_ptr<TerraSolverWarping>(new TerraSolverWarping(m_image.getWidth(), m_image.getHeight(), "ImageWarpingAD.t", "LMGPU"));
        }

		if (m_params.useCeres) {
			m_warpingSolverCeres = std::unique_ptr<CeresSolverWarping>(new CeresSolverWarping(m_image.getWidth(), m_image.getHeight()));
		}

	}

	void resetGPU()
	{
        OPT_FLOAT2* h_urshape = new OPT_FLOAT2[m_image.getWidth()*m_image.getHeight()];
        OPT_FLOAT*  h_mask = new OPT_FLOAT[m_image.getWidth()*m_image.getHeight()];

		for (unsigned int y = 0; y < m_image.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_image.getWidth(); x++)
			{
                h_urshape[y*m_image.getWidth() + x] = { (OPT_FLOAT)x, (OPT_FLOAT)y };
                h_mask[y*m_image.getWidth() + x] = (OPT_FLOAT)m_imageMask(x, y);
			}
		}

		setConstraintImage(1.0f);

        cutilSafeCall(cudaMemcpy(d_urshape, h_urshape, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy(d_warpField, h_urshape, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemset(d_warpAngles, 0, sizeof(OPT_FLOAT)*m_image.getWidth()*m_image.getHeight()));
        cutilSafeCall(cudaMemcpy(d_mask, h_mask, sizeof(OPT_FLOAT)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_urshape;
		delete h_mask;
	}

	void setConstraintImage(float alpha)
	{
        OPT_FLOAT2* h_constraints = new OPT_FLOAT2[m_image.getWidth()*m_image.getHeight()];
		//printf("m_constraints.size() = %d\n", m_constraints.size());
		for (unsigned int y = 0; y < m_image.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_image.getWidth(); x++)
			{
                h_constraints[y*m_image.getWidth() + x] = { (OPT_FLOAT)-1.0, (OPT_FLOAT)-1.0 };
			}
		}

		for (unsigned int k = 0; k < m_constraints.size(); k++)
		{
			int x = m_constraints[k][0];
			int y = m_constraints[k][1];

			if (m_imageMask(x, y) == 0)
			{
                OPT_FLOAT newX = (1.0f - alpha)*(float)x + alpha*(float)m_constraints[k][2];
                OPT_FLOAT newY = (1.0f - alpha)*(float)y + alpha*(float)m_constraints[k][3];


                h_constraints[y*m_image.getWidth() + x] = { newX, newY };
			}
		}



		cutilSafeCall(cudaMemcpy(d_constraints, h_constraints, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyHostToDevice));
		delete h_constraints;
	}

	~ImageWarping() {
		cutilSafeCall(cudaFree(d_urshape));
		cutilSafeCall(cudaFree(d_warpField));
		cutilSafeCall(cudaFree(d_constraints));
		cutilSafeCall(cudaFree(d_warpAngles));
	}

	ColorImageR32G32B32* solve() {
		float weightFit = 100.0f;
		float weightReg = 0.01f;



		if (m_params.useCUDA) {
			resetGPU();

            std::cout << std::endl << std::endl;
            for (unsigned int i = 1; i < m_params.numIter; i++)	{
				std::cout << "//////////// ITERATION" << i << "  (CUDA) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)m_params.numIter);
                assert(!OPT_DOUBLE_PRECISION);
                m_warpingSolver->solveGN((float2*)d_urshape, (float2*)d_warpField, (float*)d_warpAngles, (float2*)d_constraints, (float*)d_mask, m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg);
                if (i == 1 && m_params.earlyOut) break;
				//	std::cout << std::endl;
			}
			copyResultToCPU();
		}

        if (m_params.useTerra) {
			resetGPU();

			std::cout << std::endl << std::endl;

            for (unsigned int i = 1; i < m_params.numIter; i++)	{
				std::cout << "//////////// ITERATION" << i << "  (TERRA) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)m_params.numIter);

                m_warpingSolverTerra->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, m_params.nonLinearIter, m_params.linearIter, m_params.patchIter, weightFit, weightReg, m_optGNIters);
                if (i == 1 && m_params.earlyOut) break;
				//	std::cout << std::endl;
			}
			copyResultToCPU();
		}


        if (m_params.useOpt) {
			resetGPU();

			std::cout << std::endl << std::endl;

            for (unsigned int i = 1; i < m_params.numIter; i++)	{

				std::cout << "//////////// ITERATION" << i << "  (DSL AD) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)m_params.numIter);

                m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, m_params.nonLinearIter, m_params.linearIter, m_params.patchIter, weightFit, weightReg, m_optGNIters);
				std::cout << std::endl;
                if (i == 1 && m_params.earlyOut) break;
			}

			copyResultToCPU();
		}

        if (m_params.useOptLM) {
            resetGPU();

            std::cout << std::endl << std::endl;

            for (unsigned int i = 1; i < m_params.numIter; i++)	{

                std::cout << "//////////// ITERATION" << i << "  (DSL LM AD) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)m_params.numIter);

                m_warpingSolverTerraLMAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, m_params.nonLinearIter, m_params.linearIter, m_params.patchIter, weightFit, weightReg, m_optLMIters);
                std::cout << std::endl;
                if (i == 1 && m_params.earlyOut) break;
            }

            copyResultToCPU();
        }

        if (m_params.useCeres) {
			resetGPU();

			const int pixelCount = m_image.getWidth()*m_image.getHeight();

            OPT_FLOAT2* h_warpField = new OPT_FLOAT2[pixelCount];
            OPT_FLOAT* h_warpAngles = new OPT_FLOAT[pixelCount];

            OPT_FLOAT2* h_urshape = new OPT_FLOAT2[pixelCount];
            OPT_FLOAT*  h_mask = new OPT_FLOAT[pixelCount];
            OPT_FLOAT2* h_constraints = new OPT_FLOAT2[pixelCount];

			float totalCeresTimeMS = 0.0f;

            cutilSafeCall(cudaMemcpy(h_urshape, d_urshape, sizeof(OPT_FLOAT2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_mask, d_mask, sizeof(OPT_FLOAT) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(OPT_FLOAT2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(OPT_FLOAT2) * pixelCount, cudaMemcpyDeviceToHost));
            cutilSafeCall(cudaMemcpy(h_warpAngles, d_warpAngles, sizeof(OPT_FLOAT) * pixelCount, cudaMemcpyDeviceToHost));

			std::cout << std::endl << std::endl;

            for (unsigned int i = 1; i < m_params.numIter; i++)	{
				std::cout << "//////////// ITERATION" << i << "  (CERES) ///////////////" << std::endl;
                setConstraintImage((float)i / (float)m_params.numIter);
                cutilSafeCall(cudaMemcpy(h_constraints, d_constraints, sizeof(OPT_FLOAT2) * pixelCount, cudaMemcpyDeviceToHost));

                totalCeresTimeMS = m_warpingSolverCeres->solve(h_warpField, h_warpAngles, h_urshape, h_constraints, h_mask, weightFit, weightReg, m_ceresIters);
                std::cout << std::endl;
                if (i == 1 && m_params.earlyOut) break;
			}

            cutilSafeCall(cudaMemcpy(d_warpField, h_warpField, sizeof(OPT_FLOAT2) * pixelCount, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_warpAngles, h_warpAngles, sizeof(OPT_FLOAT) * pixelCount, cudaMemcpyHostToDevice));
			copyResultToCPU();

			std::cout << "Ceres time for final iteration: " << totalCeresTimeMS << "ms" << std::endl;

			//std::cout << "testing CERES cost function by calling AD..." << std::endl;

			//m_warpingSolverTerraAD->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, nonLinearIter, linearIter, patchIter, weightFit, weightReg);
            //m_warpingSolverCeres->solve(d_warpField, d_warpAngles, d_urshape, d_constraints, d_mask, weightFit, weightReg, m_ceresIters);
			std::cout << std::endl;
		}


   
        std::string resultSuffix = OPT_DOUBLE_PRECISION ? "_double" : "_float";

        resultSuffix += std::to_string(m_image.getWidth());
        saveSolverResults("results/", resultSuffix, m_ceresIters, m_optGNIters, m_optLMIters);
        
        double optCost = m_params.useOpt ? m_warpingSolverTerraAD->finalCost()  : nan(nullptr);
        double optLMCost = m_params.useOptLM ? m_warpingSolverTerraLMAD->finalCost() : nan(nullptr);
        double ceresCost = m_params.useCeres ? m_warpingSolverCeres->finalCost() : nan(nullptr);
        reportFinalCosts("Image Warping", m_params, optCost, optLMCost, ceresCost);

		return &m_resultColor;
	}

    vec2f toVec2(OPT_FLOAT2 p) {
		return vec2f(p.x, p.y);
	}

    void rasterizeTriangle(OPT_FLOAT2 p0, OPT_FLOAT2 p1, OPT_FLOAT2 p2, vec3f c0, vec3f c1, vec3f c2) {
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

		//bound
		//loop
		//point in trinagle
		// z-test?
	}

	void copyResultToCPU() {
		m_scale = 1.0f;
		m_resultColor = ColorImageR32G32B32(m_image.getWidth()*m_scale, m_image.getHeight()*m_scale);
		m_resultColor.setPixels(vec3f(255.0f, 255.0f, 255.0f));

        OPT_FLOAT2* h_warpField = new OPT_FLOAT2[m_image.getWidth()*m_image.getHeight()];
        cutilSafeCall(cudaMemcpy(h_warpField, d_warpField, sizeof(OPT_FLOAT2)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));

		unsigned int c = 3;
		for (unsigned int y = 0; y < m_image.getHeight(); y++)
		{
			for (unsigned int x = 0; x < m_image.getWidth(); x++)
			{
				if (y + 1 < m_image.getHeight() && x + 1 < m_image.getWidth())
				{
					if (m_imageMask(x, y) == 0)
					{
                        OPT_FLOAT2 pos00 = h_warpField[y*m_image.getWidth() + x];
                        OPT_FLOAT2 pos01 = h_warpField[y*m_image.getWidth() + (x + 1)];
                        OPT_FLOAT2 pos10 = h_warpField[(y + 1)*m_image.getWidth() + x];
                        OPT_FLOAT2 pos11 = h_warpField[(y + 1)*m_image.getWidth() + (x + 1)];

						vec3f v00 = m_imageColor(x, y);
						vec3f v01 = m_imageColor(x + 1, y);
						vec3f v10 = m_imageColor(x, y + 1);
						vec3f v11 = m_imageColor(x + 1, y + 1);

						bool valid00 = (m_imageMask(x, y) == 0);
						bool valid01 = (m_imageMask(x, y + 1) == 0);
						bool valid10 = (m_imageMask(x + 1, y) == 0);
						bool valid11 = (m_imageMask(x + 1, y + 1) == 0);

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
	bool m_lmOnlyFullSolve;

	OPT_FLOAT2*	d_urshape;
    OPT_FLOAT2* d_warpField;
    OPT_FLOAT2* d_constraints;
	OPT_FLOAT*  d_warpAngles;
    OPT_FLOAT*  d_mask;

    std::vector<SolverIteration> m_ceresIters;
    std::vector<SolverIteration> m_optGNIters;
    std::vector<SolverIteration> m_optLMIters;

	std::vector<std::vector<int>>& m_constraints;
    CombinedSolverParameters m_params;

	std::unique_ptr<CUDAWarpingSolver>	    m_warpingSolver;

    std::unique_ptr<TerraSolverWarping>		m_warpingSolverTerraAD;

    std::unique_ptr<TerraSolverWarping>		m_warpingSolverTerraLMAD;

    std::unique_ptr<TerraSolverWarping>		m_warpingSolverTerra;

    std::unique_ptr<CeresSolverWarping>		m_warpingSolverCeres;
};

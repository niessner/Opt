#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraSolverWarping.h"
#include "ImageHelper.h"

class ImageWarping {
public:
	ImageWarping(const ColorImageR32& sourceImage, const ColorImageR32& targetImage) {

		/*
        const unsigned int numLevels = 5;
		const float sigmas[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        */
        
        const unsigned int numLevels = 2;
        const float sigmas[2] = { 1.0f, 5.0f };
        /*
        const unsigned int numLevels = 3;
        const float sigmas[3] = { 1.0f, 3.0, 5.0f };
        */
		//const unsigned int numLevels = 1;
		//const float sigmas[numLevels] = { 4.0f };

		m_levels.resize(numLevels);
		for (unsigned int i = 0; i < numLevels; i++) {
			ColorImageR32 targetFiltered = targetImage;
			ColorImageR32 sourceFiltered = sourceImage;
			ImageHelper::filterGaussian(targetFiltered, sigmas[i]);
			ImageHelper::filterGaussian(sourceFiltered, sigmas[i]);
			LodePNG::save(ImageHelper::convertGrayToColor(targetFiltered), "target_filtered" + std::to_string(i) + ".png");
			LodePNG::save(ImageHelper::convertGrayToColor(sourceFiltered), "source_filtered" + std::to_string(i) + ".png");
			m_levels[i].init(sourceFiltered, targetFiltered);
		}		
		resetGPU();


		m_solverOpt = new TerraSolverWarping(sourceImage.getWidth(), sourceImage.getHeight(), "OpticalFlowAD.t", "gaussNewtonGPU");

	}

	void resetGPU()
	{
		for (size_t i = 0; i < m_levels.size(); i++) {
			m_levels[i].resetFlowVectors();
		}
	}


	~ImageWarping() {
		SAFE_DELETE(m_solverOpt);


	}

	BaseImage<float2> solve() {
		float weightFit = 10.0f;
		float weightReg = 0.1f;

        float fitTarget = 50.0f;

		unsigned int numRelaxIter = 3;
		unsigned int nonLinearIter = 1;
		unsigned int linearIter = 50;


        float fitStepSize = (fitTarget - weightFit) / (numRelaxIter);

		//unsigned int numIter = 20;
		//unsigned int nonLinearIter = 32;
		//unsigned int linearIter = 50;
		//unsigned int patchIter = 32;

		resetGPU();

		for (int i = (int)m_levels.size() - 1; i >= 0; i--) {
			
			if (i < (int)m_levels.size() - 1) {
				//init from previous levels if possible
				m_levels[i].initFlowVectorsFromOther(m_levels[i + 1]);
			}
			
			HierarchyLevel& level = m_levels[i];
			for (unsigned int k = 0; k < numRelaxIter; k++)  {
                weightFit += fitStepSize;
				std::cout << "//////////// ITERATION " << k << " (on hierarchy level " << i << " )  (DSL AD) ///////////////" << std::endl;
				m_solverOpt->solve(level.d_flowVectors, level.d_source, level.d_target, level.d_targetDU, level.d_targetDV, nonLinearIter, linearIter, 1, weightFit, weightReg);
				std::cout << std::endl;
			}

			
		}
		//copyResultToCPU();
		
		return m_levels[0].getFlowVectors();
	}


private:

	ColorImageR32 m_sourceImage;
	ColorImageR32 m_targetImage;
	ColorImageR32 m_result;

	struct HierarchyLevel {
		HierarchyLevel() {
			m_width = 0;
			m_height = 0;
			d_source = NULL;
			d_target = NULL;
			d_targetDU = NULL;
			d_targetDV = NULL;
			d_flowVectors = NULL;
		}

		void init(const ColorImageR32& source, const ColorImageR32& target) {
			assert(source.getWidth() == target.getWidth() && source.getHeight() == target.getHeight());
			m_width = source.getWidth();
			m_height = source.getHeight();
			CUDA_SAFE_CALL(cudaMalloc(&d_source, sizeof(float)*m_width*m_height));
			CUDA_SAFE_CALL(cudaMalloc(&d_target, sizeof(float)*m_width*m_height));
			CUDA_SAFE_CALL(cudaMalloc(&d_targetDU, sizeof(float)*m_width*m_height));
			CUDA_SAFE_CALL(cudaMalloc(&d_targetDV, sizeof(float)*m_width*m_height));
			CUDA_SAFE_CALL(cudaMalloc(&d_flowVectors, sizeof(float2)*m_width*m_height));

			ColorImageR32 targetDU = computeDU(target);
			ColorImageR32 targetDV = computeDV(target);
			BaseImage<float2> initFlowVectors(m_width, m_height);
			initFlowVectors.setPixels(make_float2(0.0f, 0.0f));

			CUDA_SAFE_CALL(cudaMemcpy(d_source, source.getData(), sizeof(float)*m_width*m_height, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_target, target.getData(), sizeof(float)*m_width*m_height, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_targetDU, targetDU.getData(), sizeof(float)*m_width*m_height, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_targetDV, targetDV.getData(), sizeof(float)*m_width*m_height, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_flowVectors, initFlowVectors.getData(), sizeof(float2)*m_width*m_height, cudaMemcpyHostToDevice));

			LodePNG::save(ImageHelper::convertGrayToColor(targetDU), "testDU.png");
			LodePNG::save(ImageHelper::convertGrayToColor(targetDV), "testDV.png");
		}

		void resetFlowVectors() {
			if (m_width == 0 || m_height == 0) return;
			BaseImage<float2> initFlowVectors(m_width, m_height);
			initFlowVectors.setPixels(make_float2(0.0f, 0.0f));
			CUDA_SAFE_CALL(cudaMemcpy(d_flowVectors, initFlowVectors.getData(), sizeof(float2)*m_width*m_height, cudaMemcpyHostToDevice));
		}

		void initFlowVectorsFromOther(const HierarchyLevel& level) {
			assert(level.m_width == m_width && level.m_height == m_height);
			CUDA_SAFE_CALL(cudaMemcpy(d_flowVectors, level.d_flowVectors, sizeof(float2)*m_height*m_width, cudaMemcpyDeviceToDevice));
		}

		void free() {
			CUDA_SAFE_CALL(cudaFree(d_source));
			CUDA_SAFE_CALL(cudaFree(d_target));
			CUDA_SAFE_CALL(cudaFree(d_targetDU));
			CUDA_SAFE_CALL(cudaFree(d_targetDV));
			CUDA_SAFE_CALL(cudaFree(d_flowVectors));
		}

		ColorImageR32 computeDU(const ColorImageR32& image) {
			ColorImageR32 res(image.getWidth(), image.getHeight());
			res.setPixels(0.0f);
			for (unsigned int j = 1; j < image.getHeight() - 1; j++) {
				for (unsigned int i = 1; i < image.getWidth() - 1; i++) {
					float d =
						- image(i - 1, j - 1) - image(i - 1, j) - image(i - 1, j + 1) 
						+ image(i + 1, j - 1) + image(i + 1, j) + image(i + 1, j + 1);
					res(i, j) = d / 8.0f;
				}
			}
			return res;
		}

		ColorImageR32 computeDV(const ColorImageR32& image) {
			ColorImageR32 res(image.getWidth(), image.getHeight());
			res.setPixels(0.0f);
			for (unsigned int j = 1; j < image.getHeight() - 1; j++) {
				for (unsigned int i = 1; i < image.getWidth() - 1; i++) {
					float d =
						- image(i - 1, j - 1) - image(i, j - 1) - image(i + 1, j - 1)
						+ image(i - 1, j + 1) + image(i, j + 1) + image(i + 1, j + 1);
					res(i, j) = d / 8.0f;
				}
			}
			return res;
		}

		BaseImage<float2> getFlowVectors() const {
			BaseImage<float2> flowVectors(m_width, m_height);
			CUDA_SAFE_CALL(cudaMemcpy(flowVectors.getData(), d_flowVectors, sizeof(float2)*m_width*m_height, cudaMemcpyDeviceToHost));
			return flowVectors;
		}

		unsigned int m_width;
		unsigned int m_height;
		float*	d_source;
		float*	d_target;
		float*	d_targetDU;
		float*	d_targetDV;
		float2*	d_flowVectors;	//unknowns
	};
	
	std::vector<HierarchyLevel> m_levels;


	TerraSolverWarping*		m_solverOpt;

};

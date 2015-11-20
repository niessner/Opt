#pragma once

#include <cuda_runtime.h> 
#include <cstdlib>
#include "cudaUtil.h"
#include <cuda_d3d11_interop.h> 
#include "cuda_SimpleMatrixUtil.h"
#include "Timer.h"
#include "CUDAHoleFiller.h"
#include "Cuda\CUDAScan.h"

#include "RGBDSensor.h"

class CUDARGBDAdapter
{
	public:

		CUDARGBDAdapter();
		~CUDARGBDAdapter();

		void OnD3D11DestroyDevice();
			
		HRESULT OnD3D11CreateDevice(ID3D11Device* device, RGBDSensor* depthSensor, unsigned int width, unsigned int height);

		HRESULT process(ID3D11DeviceContext* context);

		float* GetDepthMapResampledFloat();
		float4* GetColorMapResampledFloat4();

		//! Get the intrinsic camera matrix of the depth sensor
		const mat4f& getDepthIntrinsics() const
		{
			return m_depthIntrinsics;
		}

		const mat4f& getDepthIntrinsicsInv() const
		{
			return m_depthIntrinsicsInv;
		}

		//! Get the intrinsic camera matrix of the color sensor
		const mat4f& getColorIntrinsics() const
		{
			return m_colorIntrinsics;
		}

		const mat4f& getColorIntrinsicsInv() const
		{
			return m_colorIntrinsicsInv;
		}

		const mat4f& getDepthExtrinsics() const
		{
			return m_depthExtrinsics;
		}

		const mat4f& getDepthExtrinsicsInv() const
		{
			return m_depthExtrinsicsInv;
		}

		const mat4f& getColorExtrinsics() const
		{
			return m_colorExtrinsics;
		}

		const mat4f& getColorExtrinsicsInv() const
		{
			return m_colorExtrinsicsInv;
		}

		unsigned int getWidth() const;
		unsigned int getHeight() const;

		float4* GetColorMapFloat4()
		{
			return d_colorMapFloat4;
		}

		unsigned int getFrameNumber() const {
			return m_frameNumber;
		}

		RGBDSensor* getRGBDSensor(){
			return m_RGBDSensor;
		}

	private:
		
		RGBDSensor*	m_RGBDSensor;
		unsigned int m_frameNumber;

		mat4f m_depthIntrinsics;
		mat4f m_depthIntrinsicsInv;

		mat4f m_depthExtrinsics;
		mat4f m_depthExtrinsicsInv;

		mat4f m_colorIntrinsics;
		mat4f m_colorIntrinsicsInv;

		mat4f m_colorExtrinsics;
		mat4f m_colorExtrinsicsInv;

		unsigned int m_width;
		unsigned int m_height;

		//! depth texture float [D]
		float* d_depthMapFloat;					// float
		float* d_depthMapResampledFloat;		// resampled depth
	
		//! color texture float [R G B A]
		BYTE*	d_colorMapRaw;
		float4*	d_colorMapFloat4;
		float4* d_colorMapResampledFloat4;


		Timer m_timer;
};

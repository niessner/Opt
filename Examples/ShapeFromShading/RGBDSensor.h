#pragma once

#include <windows.h>
#include "Timer.h"
#include <cassert>
#include "Eigen.h"
#include <d3dx9math.h>


class RGBDSensor
{
	public:

		// Default constructor
		RGBDSensor();

		//! Init
		void init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight, unsigned int depthRingBufferSize);

		//! Destructor; releases allocated ressources
		virtual ~RGBDSensor();

		//! Connected to Depth Sensor
		virtual HRESULT createFirstConnected() = 0;

		//! Processes the Kinect depth data 
		virtual HRESULT processDepth() = 0;
		
		//! Processes the Kinect color data
		virtual HRESULT processColor() = 0;

		//! Toggles the near-mode if available
		virtual HRESULT toggleNearMode();

		//! Get the intrinsic camera matrix of the depth sensor
		const mat4f& getDepthIntrinsics() const;
		const mat4f& getDepthIntrinsicsInv() const;

		//! Get the intrinsic camera matrix of the depth sensor
		const mat4f& getColorIntrinsics() const;
		const mat4f& getColorIntrinsicsInv() const;
		const mat4f& getDepthExtrinsics() const;
		const mat4f& getDepthExtrinsicsInv() const;
		const mat4f& getColorExtrinsics() const;
		const mat4f& getColorExtrinsicsInv() const;

		void initializeDepthIntrinsics(float fovX, float fovY, float centerX, float centerY);
		void initializeColorIntrinsics(float fovX, float fovY, float centerX, float centerY);

		void initializeDepthExtrinsics(const Matrix3f& R, const Vector3f& t);
		void initializeColorExtrinsics(const Matrix3f& R, const Vector3f& t);
		void initializeDepthExtrinsics(const mat4f& m);
		void initializeColorExtrinsics(const mat4f& m);

		void incrementRingbufIdx();

		//! gets the pointer to depth array
		float *getDepthFloat();

		//! gets the pointer to color array
		vec4uc *getColorRGBX();

		unsigned int getColorWidth() const;
		unsigned int getColorHeight() const;
		unsigned int getDepthWidth() const;
		unsigned int getDepthHeight() const;

		void reset() {
			if (m_recordedDepthData.size()) {
				for (auto& d : m_recordedDepthData) {
					delete[] d;
				}
				m_recordedDepthData.clear();
			}
			if (m_recordedColorData.size()) {
 				for (auto& c : m_recordedColorData) {
					delete[] c;
				}
				m_recordedColorData.clear();
			}			
			
		}

		void recordFrame() {

			//memcpy(h_depth, depth, sizeof(float)*getDepthWidth()*getDepthHeight());
			//memcpy(h_color, color, sizeof(vec4uc)*getColorHeight()*getColorWidth());

			m_recordedDepthData.push_back(m_depthFloat[m_currentRingBufIdx]);
			m_recordedColorData.push_back(m_colorRGBX);

			m_depthFloat[m_currentRingBufIdx] = new float[getDepthWidth()*getDepthHeight()];
			m_colorRGBX = new vec4uc[getColorWidth()*getColorWidth()];
		}

		void saveRecordedFramesToFile(const std::string& filename) {
			if (m_recordedDepthData.size() == 0 || m_recordedColorData.size() == 0) return;

			CalibratedSensorData cs;
			cs.m_DepthImageWidth = getDepthWidth();
			cs.m_DepthImageHeight = getDepthHeight();
			cs.m_ColorImageWidth = getColorWidth();
			cs.m_ColorImageHeight = getColorHeight();
			cs.m_DepthNumFrames = (unsigned int)m_recordedDepthData.size();
			cs.m_ColorNumFrames = (unsigned int)m_recordedColorData.size();

			cs.m_CalibrationDepth.m_Intrinsic = getDepthIntrinsics();
			std::swap(cs.m_CalibrationDepth.m_Intrinsic(0,2), cs.m_CalibrationDepth.m_Intrinsic(0,3));
			std::swap(cs.m_CalibrationDepth.m_Intrinsic(1,2), cs.m_CalibrationDepth.m_Intrinsic(1,3));
			cs.m_CalibrationDepth.m_Intrinsic(2,2) = 1.0f;
			cs.m_CalibrationDepth.m_Intrinsic(1,1) *= -1.0f;
			cs.m_CalibrationDepth.m_Extrinsic = getDepthExtrinsics();
			cs.m_CalibrationDepth.m_IntrinsicInverse = cs.m_CalibrationDepth.m_Intrinsic.getInverse();
			cs.m_CalibrationDepth.m_ExtrinsicInverse = cs.m_CalibrationDepth.m_Extrinsic.getInverse();

			cs.m_CalibrationColor.m_Intrinsic = getColorIntrinsics();
			std::swap(cs.m_CalibrationColor.m_Intrinsic(0,2), cs.m_CalibrationColor.m_Intrinsic(0,3));
			std::swap(cs.m_CalibrationColor.m_Intrinsic(1,2), cs.m_CalibrationColor.m_Intrinsic(1,3));
			cs.m_CalibrationColor.m_Intrinsic(2,2) = 1.0f;
			cs.m_CalibrationColor.m_Intrinsic(1,1) *= -1.0f;
			cs.m_CalibrationColor.m_Extrinsic = getColorExtrinsics();
			cs.m_CalibrationColor.m_IntrinsicInverse = cs.m_CalibrationColor.m_Intrinsic.getInverse();
			cs.m_CalibrationColor.m_ExtrinsicInverse = cs.m_CalibrationColor.m_Extrinsic.getInverse();

			cs.m_DepthImages.resize(cs.m_DepthNumFrames);
			cs.m_ColorImages.resize(cs.m_ColorNumFrames);
			unsigned int dFrame = 0;
			for (auto& a : m_recordedDepthData) {
				cs.m_DepthImages[dFrame] = a;
				dFrame++;
			}
			unsigned int cFrame = 0;
			for (auto& a : m_recordedColorData) {
				cs.m_ColorImages[cFrame] = a;
				cFrame++;
			}

			std::cout << cs << std::endl;
			std::cout << "dumping recorded frames... ";
			BinaryDataStreamFile outStream(filename, true);
			//BinaryDataStreamZLibFile outStream(filename, true);
			outStream << cs;
			std::cout << "done" << std::endl;

			m_recordedDepthData.clear();
			m_recordedColorData.clear();	//destructor of cs frees all allocated data
		}

protected:

		unsigned int m_currentRingBufIdx;

		mat4f m_depthIntrinsics;
		mat4f m_depthIntrinsicsInv;

		mat4f m_depthExtrinsics;
		mat4f m_depthExtrinsicsInv;

		mat4f m_colorIntrinsics;
		mat4f m_colorIntrinsicsInv;

		mat4f m_colorExtrinsics;
		mat4f m_colorExtrinsicsInv;

		std::vector<float*> m_depthFloat;
		vec4uc*				m_colorRGBX;

		LONG   m_depthWidth;
		LONG   m_depthHeight;

		LONG   m_colorWidth;
		LONG   m_colorHeight;

		bool   m_bNearMode;

private:
	std::list<float*> m_recordedDepthData;
	std::list<vec4uc*>	m_recordedColorData;
};

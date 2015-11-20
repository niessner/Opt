
#include "stdafx.h"

#include "RGBDSensor.h"
#include <limits>

RGBDSensor::RGBDSensor()
{
	m_depthWidth  = 0;
	m_depthHeight = 0;

	m_colorWidth  = 0;
	m_colorHeight = 0;

	m_colorRGBX = NULL;

	m_bNearMode = false;

	m_currentRingBufIdx = 0;
}


void RGBDSensor::init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight, unsigned int depthRingBufferSize)
{
	std::cout << "sensor dimensions depth ( " << depthWidth << " / " << depthHeight <<" )" << std::endl;
	std::cout << "sensor dimensions color ( " << colorWidth << " / " << colorHeight <<" )" << std::endl;
	m_depthWidth  = static_cast<LONG>(depthWidth);
	m_depthHeight = static_cast<LONG>(depthHeight);

	m_colorWidth  = static_cast<LONG>(colorWidth);
	m_colorHeight = static_cast<LONG>(colorHeight);

	for (size_t i = 0; i < m_depthFloat.size(); i++) {
		SAFE_DELETE_ARRAY(m_depthFloat[i]);
	}
	m_depthFloat.resize(depthRingBufferSize);
	for (unsigned int i = 0; i<depthRingBufferSize; i++) {
		m_depthFloat[i] = new float[m_depthWidth*m_depthHeight];
	}
	
	SAFE_DELETE_ARRAY(m_colorRGBX);
	m_colorRGBX = new vec4uc[m_colorWidth*m_colorHeight];

	m_bNearMode = false;

}

RGBDSensor::~RGBDSensor()
{
	// done with pixel data
	SAFE_DELETE_ARRAY(m_colorRGBX);
	
	for (size_t i = 0; i < m_depthFloat.size(); i++) {
		SAFE_DELETE_ARRAY(m_depthFloat[i]);
	}
	m_depthFloat.clear();

	reset();
}

HRESULT RGBDSensor::toggleNearMode()
{
	m_bNearMode = !m_bNearMode;

	return S_OK;
}

//! Get the intrinsic camera matrix of the depth sensor
const mat4f& RGBDSensor::getDepthIntrinsics() const
{
	return m_depthIntrinsics;
}

const mat4f& RGBDSensor::getDepthIntrinsicsInv() const
{
	return m_depthIntrinsicsInv;
}

//! Get the intrinsic camera matrix of the color sensor
const mat4f& RGBDSensor::getColorIntrinsics() const
{
	return m_colorIntrinsics;
}

const mat4f& RGBDSensor::getColorIntrinsicsInv() const
{
	return m_colorIntrinsicsInv;
}

const mat4f& RGBDSensor::getDepthExtrinsics() const
{
	return m_depthExtrinsics;
}

const mat4f& RGBDSensor::getDepthExtrinsicsInv() const
{
	return m_depthExtrinsicsInv;
}

const mat4f& RGBDSensor::getColorExtrinsics() const
{
	return m_colorExtrinsics;
}

const mat4f& RGBDSensor::getColorExtrinsicsInv() const
{
	return m_colorExtrinsicsInv;
}

void RGBDSensor::initializeDepthExtrinsics(const Matrix3f& R, const Vector3f& t)
{
	m_depthExtrinsics = mat4f(	R(0, 0), R(0, 1), R(0, 2), t[0],
									R(1, 0), R(1, 1), R(1, 2), t[1],
									R(2, 0), R(2, 1), R(2, 2), t[2],
									0.0f, 0.0f, 0.0f, 1.0f);
	m_depthExtrinsicsInv = m_depthExtrinsics.getInverse();
}

void RGBDSensor::initializeColorExtrinsics(const Matrix3f& R, const Vector3f& t)
{
	m_colorExtrinsics = mat4f(	R(0, 0), R(0, 1), R(0, 2), t[0],
									R(1, 0), R(1, 1), R(1, 2), t[1],
									R(2, 0), R(2, 1), R(2, 2), t[2],
									0.0f, 0.0f, 0.0f, 1.0f);
	m_colorExtrinsicsInv = m_colorExtrinsics.getInverse();
}

void RGBDSensor::initializeDepthExtrinsics(const mat4f& m) {
	m_depthExtrinsics = m;
	m_depthExtrinsicsInv = m.getInverse();
}
void RGBDSensor::initializeColorExtrinsics(const mat4f& m) {
	m_colorExtrinsics = m;
	m_colorExtrinsicsInv = m.getInverse();
}

void RGBDSensor::initializeDepthIntrinsics(float fovX, float fovY, float centerX, float centerY)
{
	m_depthIntrinsics = mat4f(	fovX, 0.0f, 0.0f, centerX,
									0.0f, -fovY, 0.0f, centerY,
									0.0f, 0.0f, 0.0f, 0.0f,
									0.0f, 0.0f, 0.0f, 1.0f);

	m_depthIntrinsicsInv = mat4f(	1.0f/fovX, 0.0f, 0.0f, -centerX*1.0f/fovX,
										0.0f, -1.0f/fovY, 0.0f, centerY*1.0f/fovY,
										0.0f, 0.0f, 0.0f, 0.0f,
										0.0f, 0.0f, 0.0f, 1.0f);
}

void RGBDSensor::initializeColorIntrinsics(float fovX, float fovY, float centerX, float centerY)
{
	m_colorIntrinsics = mat4f(	fovX, 0.0f, 0.0f, centerX,
									0.0f, -fovY, 0.0f, centerY,
									0.0f, 0.0f, 0.0f, 0.0f,
									0.0f, 0.0f, 0.0f, 1.0f);

	m_colorIntrinsicsInv = mat4f(	1.0f/fovX, 0.0f, 0.0f, -centerX*1.0f/fovX,
										0.0f, -1.0f/fovY, 0.0f, centerY*1.0f/fovY,
										0.0f, 0.0f, 0.0f, 0.0f,
										0.0f, 0.0f, 0.0f, 1.0f);
}

float* RGBDSensor::getDepthFloat()
{
	return m_depthFloat[m_currentRingBufIdx];
}

void RGBDSensor::incrementRingbufIdx()
{
	m_currentRingBufIdx = (m_currentRingBufIdx+1)%m_depthFloat.size();
}

//! gets the pointer to color array
vec4uc* RGBDSensor::getColorRGBX()
{ 
	return m_colorRGBX;
}

unsigned int RGBDSensor::getColorWidth()  const
{
	return m_colorWidth;
}

unsigned int RGBDSensor::getColorHeight() const
{
	return m_colorHeight;
}

unsigned int RGBDSensor::getDepthWidth()  const
{
	return m_depthWidth;
}

unsigned int RGBDSensor::getDepthHeight() const
{
	return m_depthHeight;
}

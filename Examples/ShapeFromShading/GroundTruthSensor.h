#pragma once

/************************************************************************/
/* Prime sense depth camera: Warning this is highly untested atm        */
/************************************************************************/

#include "GlobalAppState.h"

#ifdef GROUNDTRUTH_SENSOR

#include "RGBDSensor.h"
#include <string.h>




class GroundTruthSensor : public RGBDSensor
{
public:

	//! Constructor; allocates CPU memory and creates handles
	GroundTruthSensor();

	//! Destructor; releases allocated ressources
	~GroundTruthSensor();

	//! Initializes the sensor
	HRESULT createFirstConnected();

	//! Processes the depth data (and color)
	HRESULT processDepth();	
	
	//! Processes the Kinect color data
	HRESULT processColor()
	{
		HRESULT hr = S_OK;
		return hr;
	}

	//! Toggles the Kinect to near-mode; default is far mode
	HRESULT toggleNearMode()
	{
		// PrimeSense is always in near mode
		return S_OK;
	}
	
	//! Toggle enable auto white balance
	HRESULT toggleAutoWhiteBalance()
	{
		HRESULT hr = S_OK;

		// TODO

		return hr;
	}
	


protected:
	
	
	//! read depth and color from files
	HRESULT readDepthAndColor(USHORT* depthD16, vec4uc* colorRGBX);
	HRESULT readDepthAndColor(float* depthFloat, vec4uc* colorRGBX);

	std::string m_DepthFilePath;
	std::string m_ColorFilePath;

	std::string m_RefinedDepthFilePath;
	std::string m_CalibFilePath;
	std::string m_MaskFilePath;

	int m_Frameind;
		

	// to prevent drawing until we have data for both streams
	bool			m_bDepthRead;
	bool			m_bColorRead;	

	bool			m_bDepthImageIsUpdated;
	bool			m_bDepthImageCameraIsUpdated;
	bool			m_bNormalImageCameraIsUpdated;

	bool			m_bLoadCalibFromFile;
	bool			m_bSaveMaskToFile;
};

#endif
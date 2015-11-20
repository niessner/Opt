#pragma once


/************************************************************************/
/* Reads binary dump data from .sensor files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"


#ifdef BINARY_DUMP_READER

class BinaryDumpReader : public RGBDSensor
{
public:

	//! Constructor
	BinaryDumpReader();

	//! Destructor; releases allocated ressources
	~BinaryDumpReader();

	//! initializes the sensor
	HRESULT createFirstConnected();

	//! reads the next depth frame
	HRESULT processDepth();	
	
	HRESULT processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return S_OK;
	}

	HRESULT BinaryDumpReader::toggleNearMode()	{
		return S_OK;
	}

	//! Toggle enable auto white balance
	HRESULT toggleAutoWhiteBalance() {
		return S_OK;
	}

private:
	//! deletes all allocated data
	void releaseData();

	vec4uc**		m_ColorRGBXArray;
	float**			m_DepthF32Array;
	unsigned int	m_NumFrames;
	unsigned int	m_CurrFrame;
	bool			m_bHasColorData;
};


#endif

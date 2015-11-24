
#include "stdafx.h"

#include "BinaryDumpReader.h"
#include "GlobalAppState.h"
#include "MatrixConversion.h"

#ifdef BINARY_DUMP_READER

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>



BinaryDumpReader::BinaryDumpReader()
{
	m_NumFrames = 0;
	m_CurrFrame = 0;
	m_bHasColorData = false;
	m_DepthF32Array = NULL;
	m_ColorRGBXArray = NULL;
	
}

BinaryDumpReader::~BinaryDumpReader()
{
	releaseData();
}


HRESULT BinaryDumpReader::createFirstConnected()
{
	std::string filename = GlobalAppState::get().s_filenameDump;

	std::cout << "Start loading binary dump" << std::endl;	
	BinaryDataStreamFile inputStream(filename, false);
	CalibratedSensorData sensorData;
	inputStream >> sensorData;
	std::cout << "Loading finished" << std::endl;
	std::cout << sensorData << std::endl;

    /** Hack for producing small test cases. 
    int size = 2;
    // Make sure file doesn't exist or this will just append uselessly to its end
    BinaryDataStreamFile outputStream("smalltest.sensor", false);
    sensorData.m_ColorImages.resize(size);
    sensorData.m_DepthImages.resize(size);
    sensorData.m_ColorImagesTimeStamps.resize(size);
    sensorData.m_DepthImagesTimeStamps.resize(size);
    sensorData.m_DepthNumFrames = size;
    sensorData.m_ColorNumFrames = size;
    outputStream << sensorData;
    outputStream.closeStream();
    exit(0);
    */
    
    
    

	RGBDSensor::init(sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, std::max(sensorData.m_ColorImageWidth,1u), std::max(sensorData.m_ColorImageHeight,1u), 1);	
	initializeDepthIntrinsics(sensorData.m_CalibrationDepth.m_Intrinsic(0,0), sensorData.m_CalibrationDepth.m_Intrinsic(1,1), sensorData.m_CalibrationDepth.m_Intrinsic(0,2), sensorData.m_CalibrationDepth.m_Intrinsic(1,2));	
	initializeColorIntrinsics(sensorData.m_CalibrationColor.m_Intrinsic(0,0), sensorData.m_CalibrationColor.m_Intrinsic(1,1), sensorData.m_CalibrationColor.m_Intrinsic(0,2), sensorData.m_CalibrationColor.m_Intrinsic(1,2));

	initializeDepthExtrinsics(sensorData.m_CalibrationDepth.m_Extrinsic);
	initializeColorExtrinsics(sensorData.m_CalibrationColor.m_Extrinsic);

	m_NumFrames = sensorData.m_DepthNumFrames;
	assert(sensorData.m_ColorNumFrames == sensorData.m_DepthNumFrames || sensorData.m_ColorNumFrames == 0);		
	releaseData();
	m_DepthF32Array = new float*[m_NumFrames];
	for (unsigned int i = 0; i < m_NumFrames; i++) {
		m_DepthF32Array[i] = new float[getDepthWidth()*getDepthHeight()];
		for (unsigned int k = 0; k < getDepthWidth()*getDepthHeight(); k++) {			
			if (sensorData.m_DepthImages[i][k] == 0.0f) {
				m_DepthF32Array[i][k] = -std::numeric_limits<float>::infinity();
			} else {
				m_DepthF32Array[i][k] = sensorData.m_DepthImages[i][k];
			}
		}
	}

	std::cout << "loading depth done" << std::endl;
	if (sensorData.m_ColorImages.size() > 0) {
		m_bHasColorData = true;
		m_ColorRGBXArray = new vec4uc*[m_NumFrames];
		for (unsigned int i = 0; i < m_NumFrames; i++) {
			m_ColorRGBXArray[i] = sensorData.m_ColorImages[i];
			sensorData.m_ColorImages[i] = NULL;				
		}
	} else {
		m_bHasColorData = false;
	}
	sensorData.deleteData();

	std::cout << "loading color done" << std::endl;


	return S_OK;
}

HRESULT BinaryDumpReader::processDepth()
{
	if(m_CurrFrame >= m_NumFrames)
	{
		GlobalAppState::get().s_playData = false;
		std::cout << "binary dump sequence complete - press space to run again" << std::endl;
		m_CurrFrame = 0;
	}

	if(GlobalAppState::get().s_playData) {
		float* depth = getDepthFloat();
		memcpy(depth, m_DepthF32Array[m_CurrFrame], sizeof(float)*getDepthWidth()*getDepthHeight());
		incrementRingbufIdx();

		if (m_bHasColorData) {
			memcpy(m_colorRGBX, m_ColorRGBXArray[m_CurrFrame], sizeof(vec4uc)*getColorWidth()*getColorHeight());			
		}

		m_CurrFrame++;
		return S_OK;
	} else {
		return S_FALSE;
	}
}

void BinaryDumpReader::releaseData()
{
	for (unsigned int i = 0; i < m_NumFrames; i++) {
		if (m_DepthF32Array)	SAFE_DELETE_ARRAY(m_DepthF32Array[i]);
		if (m_ColorRGBXArray)	SAFE_DELETE_ARRAY(m_ColorRGBXArray[i]);
	}
	SAFE_DELETE_ARRAY(m_DepthF32Array);
	SAFE_DELETE_ARRAY(m_ColorRGBXArray);
	m_CurrFrame = 0;
	m_bHasColorData = false;
}



#endif

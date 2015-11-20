#pragma once

//Only working with Kinect 2.0 SDK (which wants to run on Win8)

#include "GlobalAppState.h"

#ifdef KINECT_ONE

#include <Kinect.h>
#include "RGBDSensor.h"

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL) {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

class KinectOneSensor : public RGBDSensor
{
public:

	KinectOneSensor()
	{
		const unsigned int colorWidth  = 1920;
		const unsigned int colorHeight = 1080;

		const unsigned int depthWidth  = 512;
		const unsigned int depthHeight = 424;

		RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight, 2);

		m_pKinectSensor = NULL;

		HRESULT hr = createFirstConnected();
		if (hr != S_OK)	throw MLIB_EXCEPTION("failed to initialize kinect");

		// create heap storage for color pixel data in RGBX format
		m_pColorRGBX = new RGBQUAD[colorWidth*colorHeight];
	
		IMultiSourceFrame* pMultiSourceFrame = NULL;
		IDepthFrame* pDepthFrame = NULL;
		IColorFrame* pColorFrame = NULL;

		hr = S_FALSE;
		while(hr != S_OK) hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

		if(SUCCEEDED(hr))
		{
			IDepthFrameReference* pDepthFrameReference = NULL;

			hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			}

			SafeRelease(pDepthFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			IColorFrameReference* pColorFrameReference = NULL;

			hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			}

			SafeRelease(pColorFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			INT64 nDepthTime = 0;
			IFrameDescription* pDepthFrameDescription = NULL;
			int nDepthWidth = 0;
			int nDepthHeight = 0;

			IFrameDescription* pColorFrameDescription = NULL;
			int nColorWidth = 0;
			int nColorHeight = 0;
			RGBQUAD *pColorBuffer = NULL;

			// get depth frame data
			hr = pDepthFrame->get_RelativeTime(&nDepthTime);

			if (SUCCEEDED(hr))
			{
				// Intrinsics
				initializeDepthIntrinsics( 3.6214298524455461e+002, 3.6220435291479595e+002, 2.5616216259758841e+002, 2.0078875999601487e+002);
				initializeColorIntrinsics(1.0556311615223119e+003, 1.0557253330803749e+003, 9.4264212485622727e+002, 5.3125563902269801e+002);

				// Extrinsics
				Matrix3f R; R.setIdentity(); Vector3f t; t.setZero();
				initializeColorExtrinsics(R, t);

				R(0, 0) = 9.9998621443730407e-001; R(0, 1) = -1.2168971895208723e-003; R(0, 2) = -5.1078465697614612e-003;
				R(1, 0) = 1.2178529255848945e-003; R(1, 1) =  9.9999924148788211e-001; R(1, 2) =  1.8400519565076159e-004;
				R(2, 0) = 5.1076187799924972e-003; R(2, 1) = -1.9022326492404629e-004; R(2, 2) =  9.9998693793744509e-001;
				
				t[0] = -5.1589449841384898e+001;  t[1] = -1.1102720138477913e+000; t[2] = -1.2127048071059605e+001;
				t[0] /= 1000.0f; t[1] /= 1000.0f; t[2] /= 1000.0f;

				initializeDepthExtrinsics(R, t);
			}
			SafeRelease(pDepthFrameDescription);
			SafeRelease(pColorFrameDescription);
		}

		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pMultiSourceFrame);
	}

	~KinectOneSensor()
	{
		if (m_pKinectSensor)			m_pKinectSensor->Release();
		if (m_pMultiSourceFrameReader)	m_pMultiSourceFrameReader->Release();
		if (m_pCoordinateMapper)		m_pCoordinateMapper->Release();
		if (m_pColorCoordinates)		delete [] m_pColorCoordinates;

		if (m_depthSpacePoints)			delete [] m_depthSpacePoints;
		if (m_cameraSpacePoints)		delete [] m_cameraSpacePoints;

		if (m_pColorRGBX)
		{
			delete [] m_pColorRGBX;
			m_pColorRGBX = NULL;
		}
	}

	HRESULT createFirstConnected()
	{
		HRESULT hr;

		hr = GetDefaultKinectSensor(&m_pKinectSensor);
		if (FAILED(hr))
		{
			return hr;
		}

		if (m_pKinectSensor)
		{
			// Initialize the Kinect and get coordinate mapper and the frame reader
			if (SUCCEEDED(hr))
			{
				hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
			}

			hr = m_pKinectSensor->Open();

			if (SUCCEEDED(hr))
			{
				hr = m_pKinectSensor->OpenMultiSourceFrameReader(
					FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
					&m_pMultiSourceFrameReader);
			}
		}

		return hr;
	}

	HRESULT processDepth()
	{
		IMultiSourceFrame* pMultiSourceFrame = NULL;
		IDepthFrame* pDepthFrame = NULL;
		IColorFrame* pColorFrame = NULL;

		HRESULT hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

		if(SUCCEEDED(hr))
		{
			IDepthFrameReference* pDepthFrameReference = NULL;

			hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
			}

			SafeRelease(pDepthFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			IColorFrameReference* pColorFrameReference = NULL;

			hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
			if (SUCCEEDED(hr))
			{
				hr = pColorFrameReference->AcquireFrame(&pColorFrame);
			}

			SafeRelease(pColorFrameReference);
		}

		if (SUCCEEDED(hr))
		{
			INT64 nDepthTime = 0;
			IFrameDescription* pDepthFrameDescription = NULL;
			
			UINT nDepthBufferSize = 0;
			UINT16 *pDepthBuffer = NULL;

			IFrameDescription* pColorFrameDescription = NULL;
		
			ColorImageFormat imageFormat = ColorImageFormat_None;
			UINT nColorBufferSize = 0;
			RGBQUAD *pColorBuffer = NULL;

			// get depth frame data
			if (SUCCEEDED(hr)) hr = pDepthFrame->get_RelativeTime(&nDepthTime);
			if (SUCCEEDED(hr)) hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
			if (SUCCEEDED(hr)) hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);

			// get color frame data
			if (SUCCEEDED(hr)) hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
			if (SUCCEEDED(hr)) hr = pColorFrame->get_RawColorImageFormat(&imageFormat);

			if (SUCCEEDED(hr))
			{
				if (m_pColorRGBX)
				{
					pColorBuffer = m_pColorRGBX;
					nColorBufferSize = m_colorWidth * m_colorHeight * sizeof(RGBQUAD);
					hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Rgba);
				}
				else
				{
					hr = E_FAIL;
				}
			}

			if (SUCCEEDED(hr))
			{	
				// Make sure we've received valid data
				if (pDepthBuffer && pColorBuffer)
				{
					#pragma omp parallel for
					for(int i = 0; i<(int)(m_depthWidth * m_depthHeight); i++)
					{
						float dF = (float)pDepthBuffer[i]*0.001f;
						if(dF >= GlobalAppState::get().s_minDepth && dF <= GlobalAppState::get().s_maxDepth) getDepthFloat()[i] = dF;
						else																				 getDepthFloat()[i] = -std::numeric_limits<float>::infinity();
					}
					incrementRingbufIdx();
					
					#pragma omp parallel for
					for(int i = 0; i<(int)(m_colorWidth * m_colorHeight); i++)
					{	
							RGBQUAD q = m_pColorRGBX[i];
							m_colorRGBX[i].x = (unsigned char)(q.rgbRed);
							m_colorRGBX[i].y = (unsigned char)(q.rgbGreen);
							m_colorRGBX[i].z = (unsigned char)(q.rgbBlue);
							m_colorRGBX[i].w = (unsigned char)(255);
					}
				}
			}

			SafeRelease(pDepthFrameDescription);
			SafeRelease(pColorFrameDescription);
		}

		SafeRelease(pDepthFrame);
		SafeRelease(pColorFrame);
		SafeRelease(pMultiSourceFrame);

		return hr;
	}

	HRESULT processColor() {
		HRESULT hr = S_OK;
		return hr;
	}
	
	HRESULT toggleAutoWhiteBalance() {
		HRESULT hr = S_OK;
		return hr;
	}

private:

	 // Current Kinect
    IKinectSensor*          m_pKinectSensor;

	RGBQUAD*                m_pColorRGBX;

	// Frame reader
	IMultiSourceFrameReader* m_pMultiSourceFrameReader;

	// Mapping
	ICoordinateMapper*		 m_pCoordinateMapper; 
	ColorSpacePoint*         m_pColorCoordinates; 
		
	unsigned int m_depthPointCount;
	DepthSpacePoint* m_depthSpacePoints;
	CameraSpacePoint* m_cameraSpacePoints;
};

#endif

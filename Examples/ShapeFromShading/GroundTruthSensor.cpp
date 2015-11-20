#include "stdafx.h"

#include "GroundTruthSensor.h"

#ifdef GROUNDTRUTH_SENSOR

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

GroundTruthSensor::GroundTruthSensor()
{
	//default HD resolution
	unsigned int depthWidth = 1920;
	unsigned int depthHeight = 1088;

	unsigned int colorWidth = 1920;
	unsigned int colorHeight = 1088;

	


	m_bDepthRead = false;
	m_bColorRead = false;
	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;
	m_bLoadCalibFromFile = false;


	//! setting the path and parameter for input data from files
	depthWidth = 1280;
	depthHeight = 1024;
	colorWidth = 1280;
	colorHeight = 1024;
	m_ColorFilePath = "D:/RealData/chenglei/images/rgb_%04d.png";
	m_DepthFilePath = "D:/RealData/chenglei/exr/depth%04d.exr";
	m_RefinedDepthFilePath = "D:/RealData/chenglei/ours/impdep%04d.exr";
	m_CalibFilePath = "D:/RealData/chenglei/calib.txt";
	m_MaskFilePath = "D:/RealData/chenglei/mask/mask_%04d.png";
	m_bLoadCalibFromFile = true;
	m_bSaveMaskToFile = true;

	m_Frameind = 0;
		


	RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight, 1);
	
}




GroundTruthSensor::~GroundTruthSensor()
{

}


HRESULT GroundTruthSensor::createFirstConnected()
{
	HRESULT hr = S_OK;

	std::cout<<"createFirstConnected"<<std::endl;


	int depthWidth, depthHeight;
	int colorWidth, colorHeight;

	char filename[256];
	
	//read in first frame of depth
	sprintf(filename,m_DepthFilePath.c_str(),m_Frameind);	
	std::cout<<filename<<std::endl;	
	cv::Mat depthimg = cv::imread(filename, 0);
	depthHeight = depthimg.rows;
	depthWidth	= depthimg.cols;
	
	sprintf(filename,m_ColorFilePath.c_str(),m_Frameind);	
	std::cout<<filename<<std::endl;
	
	cv::Mat img = cv::imread(filename);	
	colorHeight = img.rows;
	colorWidth	= img.cols;	

	if (depthWidth != getDepthWidth() || depthHeight != getDepthHeight() || colorWidth != getColorWidth() || colorHeight != getColorHeight() || depthimg.channels()!=1 )
	{
		std::cout << depthWidth << " " << depthHeight << " " << colorWidth << " " << colorHeight << "" << depthimg.channels() << std::endl;
		std::cout << "Error - expect color and depth to have different resolutions: " << std::endl;

		return S_FALSE;
	}

	float focalLengthX = 10103.35548f;
	float focalLengthY = 10122.13667f;

	float principalPosX = 959.5f;
	float principalPosY = 543.5f;

	//! if the calibration file is provided, load it
	if(m_bLoadCalibFromFile)
	{
		FILE *pcalibfile = fopen(m_CalibFilePath.c_str(),"r");
		Matrix3f tmpmat;
		fscanf(pcalibfile,"%f\n",&tmpmat(0,0));
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
				fscanf(pcalibfile,"%f ",&tmpmat(i,j));
		}
		fclose(pcalibfile);

		focalLengthX = tmpmat(0,0);
		focalLengthY = tmpmat(1,1);
		principalPosX = tmpmat(0,2);
		principalPosY = tmpmat(1,2);	
	}

	initializeDepthIntrinsics(focalLengthX, focalLengthY, principalPosX, principalPosY); // load this from file! //TODO CHECK THIS
	initializeColorIntrinsics(focalLengthX, focalLengthY, principalPosX, principalPosY); // load this from file! //TODO CHECK THIS

	//assuming depth and color images are aligned, otherwise need modification of R ant t
	Matrix3f R; R.setIdentity(); Vector3f t; t.setZero();
	initializeColorExtrinsics(R, t); // load this from file!
	initializeDepthExtrinsics(R, t); // load this from file!

	return hr;
}


HRESULT GroundTruthSensor::processDepth()
{
	HRESULT hr = S_OK;

	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;
		
	hr = readDepthAndColor(getDepthFloat(), m_colorRGBX);

	m_Frameind += 1;	

	m_bDepthImageIsUpdated = true;
	m_bDepthImageCameraIsUpdated = true;
	m_bNormalImageCameraIsUpdated = true;

	m_bDepthRead = true;
	m_bColorRead = true;

	return hr;
}


HRESULT GroundTruthSensor::readDepthAndColor(float* depthFloat, vec4uc* colorRGBX)
{
	HRESULT hr = S_OK;

	char filename[256];
	sprintf(filename,m_DepthFilePath.c_str(),m_Frameind);	
	std::cout<< "Reading Depth Image "<< filename <<std::endl;
	cv::Mat depthimg = cv::imread(filename, 0);
	if(! depthimg.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the depth" << std::endl ;
        return S_FALSE;
    }

	sprintf(filename,m_ColorFilePath.c_str(),m_Frameind);	
	std::cout<< "Reading Color Image "<< filename <<std::endl;
	cv::Mat img = cv::imread(filename);
	if(! img.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return S_FALSE;
    }

	assert(depthimg.channels()==1);
	assert(img.rows == depthimg.rows);
	assert(img.cols == depthimg.cols);
		
	float *p_depdata = depthimg.ptr<float>(0);
	#pragma omp parallel for
	for(int i=0;i<m_depthHeight*m_depthWidth;i++){
			int tmpind = i;
			float dF = (float)p_depdata[tmpind]*0.001f;
			
			if(dF >= GlobalAppState::get().s_minDepth && dF <= GlobalAppState::get().s_maxDepth) {
				depthFloat[tmpind] = dF;				
			}
			else
				depthFloat[tmpind] = -std::numeric_limits<float>::infinity();
	}
	incrementRingbufIdx();

	unsigned char *p_imgdata = img.ptr<unsigned char>(0);
	#pragma omp parallel for
	for(int i=0;i<m_colorHeight*m_colorWidth;i++){
			int tmpind = i;			
			m_colorRGBX[tmpind].x	= p_imgdata[tmpind*3];		
			m_colorRGBX[tmpind].y	= p_imgdata[tmpind*3+1];		
			m_colorRGBX[tmpind].z	= p_imgdata[tmpind*3+2];		
			m_colorRGBX[tmpind].w	= 255;
	}	
	
	return hr;
}

#endif

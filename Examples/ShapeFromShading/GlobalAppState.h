#pragma once

//#define KINECT_ONE
#define OPEN_NI
//#define GROUNDTRUTH_SENSOR
#define BINARY_DUMP_READER


#include "DXUT.h"
#include "Eigen.h"
#include "ConvergenceAnalysis.h"

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>


#define RENDERMODE_INTEGRATE 0 
#define RENDERMODE_VIEW 1

#define X_GLOBAL_APP_STATE_FIELDS \
	X(unsigned int, s_sensorIdx) \
	X(float, s_minDepth) \
	X(float, s_maxDepth) \
	X(float, s_texture_threshold) \
	X(unsigned int, s_windowWidth) \
	X(unsigned int, s_windowHeight) \
	X(unsigned int, s_RenderMode) \
	X(float, s_depthSigmaD) \
	X(float, s_depthSigmaR) \
	X(bool, s_depthFilter) \
	X(float, s_colorSigmaD) \
	X(float, s_colorSigmaR) \
	X(bool, s_colorFilter) \
	X(bool, s_bRenderModeChanged) \
	X(bool, s_bFilterKinectInputData) \
	X(bool, s_refineForeground) \
	X(bool, s_useColorForRendering) \
	X(bool, s_useSyntheticLighting) \
	X(float, s_lightingSolverThresDepth) \
	X(bool, s_playData) \
	X(std::string, s_filenameDump) \
	X(vec4f, s_ambientLightColor) \
	X(vec4f, s_diffuseLightColor) \
	X(vec4f, s_specularLightColor) \
	X(float, s_materialShininess) \
	X(vec4f, s_materialSpecular) \
	X(vec3f, s_lightDirection) \
	X(vec4f, s_materialDiffuse) \
	X(float, s_renderingDepthDiscontinuityThresOffset) \
	X(float, s_renderingDepthDiscontinuityThresLin) \
	X(float, s_remappingDepthDiscontinuityThresOffset) \
	X(float, s_remappingDepthDiscontinuityThresLin) \
	X(int, s_rowOffset) \
	X(float, s_weightPriorLight) \
	X(float, s_segmentationDepthThres) \
	X(bool, s_bUseCameraCalibration) \
	X(bool, s_onlySaveForeground) \
	X(float, s_edgeThresholdSaveToFileOffset) \
	X(float, s_edgeThresholdSaveToFileLin) \
	X(unsigned int, s_adapterWidth) \
	X(unsigned int, s_adapterHeight) \
	X(bool, s_timingsDetailledEnabled) \
	X(bool, s_timingsTotalEnabled) \
	X(unsigned int, s_numHierarchyLevels) \
    X(int, s_optimizer) \
    X(bool, s_useBlockSolver) \
	X(std::vector<unsigned int>, s_nNonLinearIterations) \
	X(std::vector<unsigned int>, s_nLinearIterations) \
	X(std::vector<unsigned int>, s_nPatchIterations) \
	X(std::vector<float>, s_weightFitting) \
	X(std::vector<float>, s_weightShadingIncrement) \
	X(std::vector<float>, s_weightRegularizer) \
	X(std::vector<float>, s_weightShadingStart) \
	X(std::vector<float>, s_weightBoundary) \
	X(std::vector<float>, s_weightPrior) \
	X(std::vector<unsigned int>, s_optimizationWidthAtLevel) \
	X(std::vector<unsigned int>, s_optimizationHeightAtLevel) \
	X(bool, s_bRecordDepthData) \
	X(bool, s_convergenceAnalysisIsRunning) \
	X(bool, s_renderToFile) \
	X(unsigned int, s_renderToFileResWidth) \
	X(unsigned int, s_renderToFileResHeight) \
	X(std::string, s_renderToFileDirectory)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif

#define checkSizeArray(a, d)( (((sizeof a)/(sizeof a[0])) >= d))

class GlobalAppState
{
public:

#define X(type, name) type name;
	X_GLOBAL_APP_STATE_FIELDS
#undef X

		//! sets the parameter file and reads
	void readMembers(const ParameterFile& parameterFile) {
		s_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_APP_STATE_FIELDS
#undef X


		unsigned int optimizationWidthAtLevel[]	 = {s_adapterWidth,  s_adapterWidth/2,  s_adapterWidth/4};
		unsigned int optimizationHeightAtLevel[] = {s_adapterHeight, s_adapterHeight/2, s_adapterHeight/4};

		if(		(s_nNonLinearIterations.size() < s_numHierarchyLevels)
			||	(s_nLinearIterations.size() < s_numHierarchyLevels)
			||	(s_nPatchIterations.size() < s_numHierarchyLevels)
			||	(s_weightFitting.size() < s_numHierarchyLevels)
			||	(s_weightShadingIncrement.size() < s_numHierarchyLevels)
			||	(s_weightRegularizer.size() < s_numHierarchyLevels)
			||	(s_weightShadingStart.size() < s_numHierarchyLevels)
			||	(s_weightBoundary.size() < s_numHierarchyLevels) 
			||	(s_weightPrior.size() < s_numHierarchyLevels)
			||	!checkSizeArray(optimizationWidthAtLevel, s_numHierarchyLevels) 
			||	!checkSizeArray(optimizationHeightAtLevel, s_numHierarchyLevels) )
		{
			//std::cout << ((sizeof nNonLinearIterations)/(sizeof nNonLinearIterations[0])) << std::endl;
			std::cout << "Number of hierarchy levels does not fit the parameters!" << std::endl; while(1);
		}

		s_optimizationWidthAtLevel.clear();
		s_optimizationHeightAtLevel.clear();
		for (unsigned int i = 0; i < s_numHierarchyLevels; i++) {
			s_optimizationWidthAtLevel.push_back(optimizationWidthAtLevel[i]);
			s_optimizationHeightAtLevel.push_back(optimizationHeightAtLevel[i]);
		}

		m_bIsInitialized = true;
	}

	void print() const {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_APP_STATE_FIELDS
#undef X
	}

	static GlobalAppState& getInstance() {
		static GlobalAppState s;
		return s;
	}
	static GlobalAppState& get() {
		return getInstance();
	}


	//! constructor
	GlobalAppState() {
		m_bIsInitialized = false;
		s_pQuery = NULL;
	}

	//! destructor
	~GlobalAppState() {

	}


	HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
	void OnD3D11DestroyDevice();

	void WaitForGPU();

	Timer	s_Timer;

	ConvergenceAnalysis<float>	s_convergenceAnalysis;

private:
	bool m_bIsInitialized;
	ParameterFile s_ParameterFile;
	ID3D11Query* s_pQuery;
};

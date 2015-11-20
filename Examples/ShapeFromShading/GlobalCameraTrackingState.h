#pragma once

/***********************************************************************************/
/* Global App state for camera tracking: reads and stores all tracking parameters  */
/***********************************************************************************/

#include "DXUT.h"

#include "stdafx.h"

#include <vector>

#define X_GLOBAL_CAMERA_APP_STATE_FIELDS \
	X(unsigned int, s_maxLevels) \
	X(std::vector<unsigned int>, s_blockSizeNormalize) \
	X(std::vector<unsigned int>, s_numBucketsNormalize) \
	X(std::vector<unsigned int>, s_localWindowSize) \
	X(std::vector<unsigned int>, s_maxOuterIter) \
	X(std::vector<unsigned int>, s_maxInnerIter) \
	X(std::vector<float>, s_distThres) \
	X(std::vector<float>, s_normalThres) \
	X(std::vector<float>, s_angleTransThres) \
	X(std::vector<float>, s_distTransThres) \
	X(std::vector<float>, s_residualEarlyOut)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif


class GlobalCameraTrackingState
{	
	public:
#define X(type, name) type name;
		X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X

		GlobalCameraTrackingState() {
			setDefault();
		}

		//! setting default parameters
		void setDefault() {
			s_blockSizeNormalize.resize(1);
			s_numBucketsNormalize.resize(1);
			s_localWindowSize.resize(1);
			s_maxOuterIter.resize(1);
			s_maxInnerIter.resize(1);
			s_distThres.resize(1);
			s_normalThres.resize(1);
			s_angleTransThres.resize(1);
			s_distTransThres.resize(1);
			s_residualEarlyOut.resize(1);

			s_maxLevels = 1;
			s_blockSizeNormalize[0] = 512;
			s_numBucketsNormalize[0] = 1024;
			s_localWindowSize[0] = 12;
			s_maxOuterIter[0] = 20;
			s_maxInnerIter[0] = 1;
			s_distThres[0] = 0.15f;
			s_normalThres[0] = 0.97f;

			s_angleTransThres[0] = 3000.0f;	// radians
			s_distTransThres[0] = 3000.4f;	// meters
			s_residualEarlyOut[0] = 0.01f;	//causes an early out if residual is smaller than this number (no early out if set to zero)
		}

		//! sets the parameter file and reads
		void readMembers(const ParameterFile& parameterFile) {
				s_ParameterFile = parameterFile;
				readMembers();
		}

		//! reads all the members from the given parameter file (could be called for reloading)
		void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
			X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X
		}

		//! prints all members
		void print() {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
			X_GLOBAL_CAMERA_APP_STATE_FIELDS
#undef X
		}

		static GlobalCameraTrackingState& getInstance() {
			static GlobalCameraTrackingState s;
			return s;
		}	
private:
	ParameterFile s_ParameterFile;
};

#include "main.h"
#include <OptImage.h>

/** \file App.cpp */
#include "App.h"


int App::launchG3DVisualizer() {
    GApp::Settings settings;
    settings.window.framed = false;
    // Change the window and other startup parameters by modifying the
    // settings class.  For example:
    settings.window.width = 1280;
    settings.window.height = 720;
    settings.window.alwaysOnTop = true;
    settings.dataDir = "../InteractiveDLL/data-files";
    _g3dVisualizer = new G3DVisualizer(settings);
    return _g3dVisualizer->run();
}

void App::init()
{
    _g3dVisualizer = NULL;
}

UINT32 App::moveWindow(int x, int y, int width, int height) {
    if (notNull(_g3dVisualizer) && _g3dVisualizer->initialized()) {
        _g3dVisualizer->sendMoveMessage(x, y, width, height);
        return 0;
    } else {
        return 2;
    }
    
}

UINT32 App::processCommand(const string &command)
{
	vector<string> words = ml::util::split(command, "\t");
	//while(words.Length() < 5) words.PushEnd("");
    if (words[0] == "load") {
        _terraFile = words[1];
    } else if (words[0] == "run") {
        if (notNull(_g3dVisualizer) && _g3dVisualizer->initialized()) {
            string optimizationMethod = words[1];
            _g3dVisualizer->sendRunOptMessage(_terraFile, optimizationMethod);
        }
        else {
            return 2;
        }
    } else if (words[0] == "check") {

        if (notNull(_g3dVisualizer) && _g3dVisualizer->initialized()) {
            _g3dVisualizer->getStatusInfo(_statusInfo);
            if (_statusInfo.informationReadSinceCompilation == false) {
                return 1;
            }
        }else {
            return 2;
        }
    }

	return 0;
}


int App::getIntegerByName(const string &s)
{
	if(s == "layerCount")
	{
		return 1;
	}
	else
	{
		MLIB_ERROR("Unknown integer");
		return -1;
	}
}


float App::getFloatByName(const string &s)
{
    
    if (s == "defineTime")
    {
        return _statusInfo.timingInfo.optDefineTime;
    }
    else if (s == "planTime")
    {
        return _statusInfo.timingInfo.optPlanTime;
    }
    else if (s == "solveTime")
    {
        return _statusInfo.timingInfo.optSolveTime;
    }
    else if (s == "solveTimeGPU")
    {
        return _statusInfo.timingInfo.optSolveTimeGPU;
    }
    else
    {
        MLIB_ERROR("Unknown integer");
        return -1;
    }
}

const char* App::getStringByName(const string &s)
{

	if(s == "terraFilename")
	{
        _queryString = _terraFile;
	}
    else if (s == "error") {
        _queryString = _statusInfo.compilerMessage;
    } else {
        MLIB_ERROR("Unknown string");
		return NULL;
	}
    return  _queryString.c_str();
}

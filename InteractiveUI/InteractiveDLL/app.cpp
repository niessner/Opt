#include "main.h"
#include <OptImage.h>
void App::init()
{
    
}

UINT32 App::processCommand(const string &command)
{
	vector<string> words = util::split(command, "\t");
	//while(words.Length() < 5) words.PushEnd("");
    _errorString = " ";
    if (words[0] == "load") {
        _terraFile = words[1];
    } else if (words[0] == "loadImage") {
        loadPNGIntoTest(words[1]);
    } else if (words[0] == "run") {
        string optimizationMethod = words[1];
        printf("%s\n", words[1].c_str());

        _optDefineTime = 0;
        _optPlanTime = 0;
        _optSolveTime = 0;
        _optSolveTimeGPU = 0;

        if (_queryBitmapInfo.colorData == NULL) {
            _errorString = "No image available";
            return -1;
        }
        for (const auto &p : _test)
        {
            _optImages[0](p.x, p.y) = 0.0;
        }
       
        uint64_t dims[] = { _test.getWidth(), _test.getHeight() };

        cudaEventCreate(&_optSolveStart);
        cudaEventCreate(&_optSolveEnd);

        ml::Timer timer;
        timer.start();

        OptState* optimizerState = Opt_NewState();
        if (optimizerState == nullptr)
        {
            _errorString = "Opt_NewState failed";
            return -1;
        }

        for (auto &image : _optImages)
            image.bind(optimizerState);

        Problem * prob = Opt_ProblemDefine(optimizerState, _terraFile.c_str(), optimizationMethod.c_str(), NULL);
        timer.stop();
        _optDefineTime = timer.getElapsedTimeMS();

        if (!prob)
        {
            _errorString = "Opt_ProblemDefine failed";
            return -1;
        }

        timer.start();
        Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims);
        timer.stop();
        _optPlanTime = timer.getElapsedTimeMS();

        if (!plan)
        {
            _errorString = "Opt_ProblemPlan failed";
            return -1;
        }

        vector<ImageBinding *> imageBindingsCPU;
        vector<ImageBinding *> imageBindingsGPU;
        for (const auto &image : _optImages)
        {
            image.syncCPUToGPU();
            imageBindingsCPU.push_back(image.terraBindingCPU);
            imageBindingsGPU.push_back(image.terraBindingGPU);
        }

        bool isGPU = ml::util::endsWith(optimizationMethod, "GPU");

        timer.start();
        if (isGPU)
        {
            cudaEventRecord(_optSolveStart);
            Opt_ProblemSolve(optimizerState, plan, imageBindingsGPU.data(), NULL);
            cudaEventRecord(_optSolveEnd);
            for (const auto &image : _optImages)
                image.syncGPUToCPU();
        }
        else {
            Opt_ProblemSolve(optimizerState, plan, imageBindingsCPU.data(), NULL);
        }
        timer.stop();
        _optSolveTime = timer.getElapsedTimeMS();
        
      
        for (int y = 0; y < _test.getHeight(); ++y) {
            for (int x = 0; x < _test.getWidth(); ++x) {
                float color = math::clamp(_optImages[0].dataCPU[y*_test.getWidth() + x], 0.0f, 255.0f);
                _result.setPixel(x, y, ml::vec4uc(color, color, color, 1));
            }
        }

        cudaEventSynchronize(_optSolveEnd);
        cudaEventElapsedTime(&_optSolveTimeGPU, _optSolveStart, _optSolveEnd);
        
    } 

	
	return 0;
}

void App::loadPNGIntoTest(const string& path) {
    _test = LodePNG::load(path);
    _result = Bitmap(ml::vec2i(_test.getWidth(), _test.getHeight()));
    for (const auto &p : _result)
        _result(p.x, p.y) = ml::vec4uc(0, 0, 0, 1);
    _optImages.resize(2);
    _optImages[0].allocate(_test.getWidth(), _test.getHeight());
    _optImages[1].allocate(_test.getWidth(), _test.getHeight());
    for (const auto &p : _test)
    {
        _optImages[0](p.x, p.y) = 0.0;
        _optImages[1](p.x, p.y) = p.value.r;
    }
}


IVBitmapInfo* App::getBitmapByName(const string &name)
{
	Bitmap *resultPtr = NULL;

    if (name == "test")
    {
        resultPtr = &_test;
    }
    else if (name == "result")
    {
        resultPtr = &_result;
    }

	if(resultPtr == NULL) return NULL;

	//resultPtr->FlipBlueAndRed();
	_queryBitmapInfo.width = resultPtr->getWidth();
    _queryBitmapInfo.height = resultPtr->getHeight();
	_queryBitmapInfo.colorData = (BYTE*)resultPtr->getPointer();
	return &_queryBitmapInfo;
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
        return _optDefineTime;
    }
    else if (s == "planTime")
    {
        return _optPlanTime;
    }
    else if (s == "solveTime")
    {
        return _optSolveTime;
    }
    else if (s == "solveTimeGPU")
    {
        return _optSolveTimeGPU;
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
        _queryString = _errorString;
    } else {
        MLIB_ERROR("Unknown string");
		return NULL;
	}
    return  _queryString.c_str();
}

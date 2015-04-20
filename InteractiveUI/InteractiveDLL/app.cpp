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
        if (_queryBitmapInfo.colorData == NULL) {
            _errorString = "No image available";
            return -1;
        }
        for (const auto &p : _test)
        {
            _optImages[0](p.x, p.y) = 0.0;
        }
       
        uint64_t dims[] = { _test.getWidth(), _test.getHeight() };

        OptState* optimizerState = Opt_NewState();
        if (optimizerState == nullptr)
        {
            _errorString = "Opt_NewState failed";
            return -1;
        }

        for (auto &image : _optImages)
            image.bind(optimizerState);

        Problem * prob = Opt_ProblemDefine(optimizerState, _terraFile.c_str(), optimizationMethod.c_str(), NULL);

        if (!prob)
        {
            _errorString = "Opt_ProblemDefine failed";
            return -1;
        }

        Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims);

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

        bool isGPU = false;//ml::util::endsWith(method.optimizerName, "GPU");

        if (isGPU)
        {
            Opt_ProblemSolve(optimizerState, plan, imageBindingsGPU.data(), NULL);
            for (const auto &image : _optImages)
                image.syncGPUToCPU();
        }
        else
            Opt_ProblemSolve(optimizerState, plan, imageBindingsCPU.data(), NULL);
      
        for (int y = 0; y < _test.getHeight(); ++y) {
            for (int x = 0; x < _test.getWidth(); ++x) {
                float color = math::clamp(_optImages[0].dataCPU[y*_test.getWidth() + x], 0.0f, 255.0f);
                _result.setPixel(x, y, ml::vec4uc(color, color, color, 1));
            }
        }
        
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

#include "main.h"

void App::init()
{
    printf("test\n");
    cout << "hello\n" << endl;
}

UINT32 App::processCommand(const string &command)
{
	/*Vector<string> words = command.Partition(" ");
	while(words.Length() < 5) words.PushEnd("");

	if (words[0] == "SynthesizeTexture") {
		SynthesizeTexture(words[1]);
	}
	else if (words[0] == "SynthesizeTextureByLayers") {
		SynthesizeTextureByLayers(words[1]);
	}
	else if (words[0] == "DeleteLayer") {
		//DeleteLayer(words[1]);
		FilterLayers(words[1]);
	}
	else if (words[0] == "ExtractVideoLayers") {
		ExtractVideoLayers();
	}
	else if (words[0] == "RBFVideoRecolor") {
		RBFVideoRecolor();
	}
	else if (words[0] == "CompareMethods") {
		CompareMethods();
	}*/
	
	return 0;
}


IVBitmapInfo* App::getBitmapByName(const string &name)
{
	/*Bitmap *resultPtr = NULL;

	if(s == "videoFrame")
	{
		if (_videocontroller.hasVideo()) {
			Bitmap *frame = _videocontroller.GetNextFrame();
			resultPtr = frame;
		}
	}
	else if (s.StartsWith("suggestFrame"))
	{
		int index = s.RemovePrefix("suggestFrame").ConvertToInteger();
		resultPtr = _videocontroller.GetSuggestionImage(index);
	}

	if(resultPtr == NULL) return NULL;

	resultPtr->FlipBlueAndRed();
	_queryBitmapInfo.width = resultPtr->Width();
	_queryBitmapInfo.height = resultPtr->Height();
	_queryBitmapInfo.colorData = (BYTE*)resultPtr->Pixels();*/
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
	if(s == "z")
	{
		_queryString = "a";
	}
	else
	{
        MLIB_ERROR("Unknown string");
		return NULL;
	}
	return _queryString.c_str();
}

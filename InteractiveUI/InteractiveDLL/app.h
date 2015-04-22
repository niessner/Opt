#include <OptImage.h>

#include <cuda_runtime.h>
class App
{
public:
    void init();
    std::mutex _lock;
    

    UINT32 processCommand(const string &command);

    IVBitmapInfo* getBitmapByName(const string &name);
    int getIntegerByName(const string &name);
    float getFloatByName(const string &name);
    const char *getStringByName(const string &name);
    void loadPNGIntoTest(const string& path);

private:

    cudaEvent_t _optSolveStart;
    cudaEvent_t _optSolveEnd;


    float _optDefineTime;
    float _optPlanTime;
    float _optSolveTime;
    float _optSolveTimeGPU;

    string _queryString;
    string _terraFile;
    string _errorString;

    IVBitmapInfo _queryBitmapInfo;
    Bitmap _result;
    Bitmap _test;

    vector<OptImage> _optImages;
};

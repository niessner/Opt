#include <OptImage.h>
class App
{
public:
    void init();

    

    UINT32 processCommand(const string &command);

    IVBitmapInfo* getBitmapByName(const string &name);
    int getIntegerByName(const string &name);
    const char *getStringByName(const string &name);
    void loadPNGIntoTest(const string& path);

private:
    string _queryString;
    string _terraFile;
    string _errorString;

    IVBitmapInfo _queryBitmapInfo;
    Bitmap _result;
    Bitmap _test;

    vector<OptImage> _optImages;
};

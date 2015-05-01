#ifndef App_h
#define App_h

#include "G3DVisualizer.h"
class App
{
private:
    string _queryString;
    string _errorString;
    string _terraFile;

    G3DVisualizer* _g3dVisualizer;

public:

    int launchG3DVisualizer();

    void init();
    std::mutex _lock;
    UINT32 moveWindow(int x, int y, int width, int height);

    UINT32 processCommand(const string &command);

    int getIntegerByName(const string &name);
    float getFloatByName(const string &name);
    const char *getStringByName(const string &name);
    void loadPNGIntoTest(const string& path);


};
#endif
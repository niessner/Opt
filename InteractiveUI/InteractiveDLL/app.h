class App
{
public:
    void init();

    UINT32 processCommand(const string &command);

    IVBitmapInfo* getBitmapByName(const string &name);
    int getIntegerByName(const string &name);
    const char *getStringByName(const string &name);
	
private:
	string _queryString;

    IVBitmapInfo _queryBitmapInfo;
    Bitmap _queryBitmapResultA;
    Bitmap _queryBitmapResultB;
    Bitmap _queryBitmapResultC;
};

#include "Main.h"

INTERACTIVEDLL_API void* IVInit()
{
    App *app = new App;

    app->init();

	return app;
}

INTERACTIVEDLL_API UINT32 IVProcessCommand(void *context, const char *s)
{
    if(context == NULL) return 1;
    App &app = *(App*)context;
    app._lock.lock();
    UINT32 result = app.processCommand(string(s));
    app._lock.unlock();
    return result;
}

INTERACTIVEDLL_API const char* IVGetStringByName(void *context, const char *s)
{
    if(context == NULL) return NULL;
    
    App &app = *(App*)context;
    app._lock.lock();
    const char* result = app.getStringByName(s);
    app._lock.unlock();
    return result;
}

INTERACTIVEDLL_API int IVGetIntegerByName(void *context, const char *s)
{
    if(context == NULL) return 0;
    App &app = *(App*)context;
    app._lock.lock();
    int result = app.getIntegerByName(s);
    app._lock.unlock();
    return result;
}

INTERACTIVEDLL_API IVBitmapInfo* IVGetBitmapByName(void *context, const char *s)
{
    if(context == NULL) return 0;
    App &app = *(App*)context;
    app._lock.lock();
    IVBitmapInfo* result = app.getBitmapByName(s);
    app._lock.unlock();
    return result;
}

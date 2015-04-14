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
    UINT32 result = app.processCommand(string(s));
    return result;
}

INTERACTIVEDLL_API const char* IVGetStringByName(void *context, const char *s)
{
    if(context == NULL) return NULL;
    App &app = *(App*)context;
    return app.getStringByName(s);
}

INTERACTIVEDLL_API int IVGetIntegerByName(void *context, const char *s)
{
    if(context == NULL) return 0;
    App &app = *(App*)context;
    return app.getIntegerByName(s);
}

INTERACTIVEDLL_API IVBitmapInfo* IVGetBitmapByName(void *context, const char *s)
{
    if(context == NULL) return 0;
    App &app = *(App*)context;
    return app.getBitmapByName(s);
}

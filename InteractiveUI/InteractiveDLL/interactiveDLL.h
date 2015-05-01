// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the BASECODEDLL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// INTERACTIVEDLL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef INTERACTIVEDLL_EXPORTS
#define INTERACTIVEDLL_API __declspec(dllexport)
#else
#define INTERACTIVEDLL_API __declspec(dllimport)
#endif

INTERACTIVEDLL_API void*         IVInit();
INTERACTIVEDLL_API UINT32        IVProcessCommand(void *context, const char *s);
INTERACTIVEDLL_API UINT32        IVRunApp(void *context);
INTERACTIVEDLL_API const char*   IVGetStringByName(void *context, const char *s);
INTERACTIVEDLL_API int           IVGetIntegerByName(void *context, const char *s);
INTERACTIVEDLL_API float         IVGetFloatByName(void *context, const char *s);
INTERACTIVEDLL_API UINT32        IVMoveWindow(void *context, int x, int y, int width, int height);

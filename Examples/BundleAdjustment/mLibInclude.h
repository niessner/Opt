
//
// mLib config options
//

#ifdef _DEBUG
#define MLIB_ERROR_CHECK
#define MLIB_BOUNDS_CHECK
#define _IDL0
#endif // _DEBUG

//
// mLib includes
//

#include <mLibCore.h>
#include <mLibLodePNG.h>
#include <mLibDepthCamera.h>
#include <mLibEigen.h>

using namespace ml;
using namespace std;

typedef ColorImageR8G8B8A8 Bitmap;
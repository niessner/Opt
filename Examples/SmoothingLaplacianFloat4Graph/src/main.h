#pragma once

#include "Resource.h"
#include "mLibInclude.h"

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
#endif

// for now, CERES only runs on release.
#ifndef _DEBUG
#define USE_CERES
#endif
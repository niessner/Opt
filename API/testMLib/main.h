
#include "mLibInclude.h"

using namespace ml;
using Bitmap = ml::ColorImageR8G8B8A8;

extern "C" {
#include "Opt.h"
}

#include "cuda_runtime.h"

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "OptImage.h"

using namespace std;

#include "testFramework.h"

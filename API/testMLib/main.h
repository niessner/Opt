
#include "mLibInclude.h"

extern "C" {
#include "Opt.h"
}

#include "cuda_runtime.h"

#include <iostream>
#include <string>
#include <vector>
#include <functional>

using namespace std;
using namespace ml;
using Bitmap = ml::ColorImageR8G8B8A8;

#include "testFramework.h"

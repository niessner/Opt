#ifndef DumpOptImage_h
#define DumpOptImage_h
#include <string>

namespace OptUtil {
    /** Assumes floating point type for now */
    void dumpOptImage(float* d_ptr, std::string filename, int width, int height, int numChannels);
}

#endif
#ifndef DumpOptImage_h
#define DumpOptImage_h
#include <string>

namespace OptUtil {
    /** datatype = 0 means float, datatype = 1 means uint8 */
    void dumpOptImage(void* d_ptr, std::string filename, int width, int height, int numChannels, int datatype = 0);
}

#endif
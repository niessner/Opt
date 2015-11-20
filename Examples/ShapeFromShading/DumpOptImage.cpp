#include "stdafx.h"
#include "DumpOptImage.h"
#include <cuda_runtime.h>
void OptUtil::dumpOptImage(float* d_ptr, std::string filename, int width, int height, int channelCount) {
    int datatype = 0; // floating point
    FILE* fileHandle = fopen(filename.c_str(), "wb"); //b for binary
    fwrite(&width, sizeof(int), 1, fileHandle);
    fwrite(&height, sizeof(int), 1, fileHandle);
    fwrite(&channelCount, sizeof(int), 1, fileHandle);
    fwrite(&datatype, sizeof(int), 1, fileHandle);
    int size = sizeof(float) *(width*height);
    void* ptr = malloc(size);
    cudaMemcpy(ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    fwrite(ptr, sizeof(float), (width*height), fileHandle);
    fclose(fileHandle);

    free(ptr);
}

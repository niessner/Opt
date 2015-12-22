#include "SimpleBuffer.h"

#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>

//using std::memcpy;



SimpleBuffer::SimpleBuffer(std::string filename, bool onGPU) :
    m_onGPU(onGPU)
{
    FILE* fileHandle = fopen(filename.c_str(), "rb"); //b for binary
    fread(&m_width,         sizeof(int), 1, fileHandle);
    fread(&m_height,        sizeof(int), 1, fileHandle);
    fread(&m_channelCount,  sizeof(int), 1, fileHandle);
    int datatype;
    fread(&datatype,        sizeof(int), 1, fileHandle);
    m_dataType = DataType(datatype);
    size_t elementSize = datatypeToSize(m_dataType);

    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr = malloc(size);
    fread(ptr, elementSize*m_channelCount, (m_width*m_height), fileHandle);
    
    fclose(fileHandle);
    if (m_onGPU) {
        cudaMalloc(&m_data, size);
        cudaMemcpy(m_data, ptr, size, cudaMemcpyHostToDevice);
        free(ptr);
    } else {
        m_data = ptr;
    }
}

SimpleBuffer::SimpleBuffer(const SimpleBuffer& other, bool onGPU) :
    m_onGPU(onGPU),
    m_width(other.m_width),
    m_height(other.m_height),
    m_channelCount(other.m_channelCount),
    m_dataType(other.m_dataType)
{
    size_t dataSize = m_width*m_height*m_channelCount*datatypeToSize(m_dataType);
    if (onGPU) {
        cudaMalloc(&m_data, dataSize);
        if (other.m_onGPU) {
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyDeviceToDevice);
        } else { 
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyHostToDevice);
        }
    } else {
        m_data = malloc(dataSize);
        if (other.m_onGPU) {
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyDeviceToHost);
        } else { // Both on CPU
            memcpy(m_data, other.m_data, dataSize);
        }
    }
}

void SimpleBuffer::save(std::string filename) {
    int datatype = m_dataType;
    FILE* fileHandle = fopen(filename.c_str(), "wb"); //b for binary
    fwrite(&m_width, sizeof(int), 1, fileHandle);
    fwrite(&m_height, sizeof(int), 1, fileHandle);
    fwrite(&m_channelCount, sizeof(int), 1, fileHandle);
    fwrite(&datatype, sizeof(int), 1, fileHandle);

    size_t elementSize = datatypeToSize(m_dataType);

    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr;
    if (m_onGPU) {
        ptr = malloc(size);
        cudaMemcpy(ptr, m_data, size, cudaMemcpyDeviceToHost);
    } else {
        ptr = m_data;
    }

    fwrite(ptr, elementSize*m_channelCount, (m_width*m_height), fileHandle);
    fclose(fileHandle);
    if (m_onGPU) {
        free(ptr);
    }
}
SimpleBuffer::~SimpleBuffer() {
    if (m_onGPU) {
        cudaFree(m_data);
    } else {
        free(m_data);
    }
}
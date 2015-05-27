#pragma  once

#ifndef OptImage_h
#define OptImage_h

#include "cuda_runtime.h"
#include <algorithm>

template<class T>
struct OptImage
{
    OptImage()
    {
        m_dataGPU = nullptr;
    }
    OptImage(int _dimX, int _dimY)
    {
        allocate(_dimX, _dimY);
    }

    void allocate(int _dimX, int _dimY)
    {
        dimX = _dimX;
        dimY = _dimY;
        m_dataCPU.resize(dimX * dimY);
        cudaMalloc(&m_dataGPU, sizeof(T) * dimX * dimY);
    }
    void syncCPUToGPU() const
    {
        cudaMemcpy(m_dataGPU, (void *)m_dataCPU.data(), sizeof(T) * dimX * dimY, cudaMemcpyHostToDevice);
    }
    void syncGPUToCPU() const
    {
        cudaMemcpy((void *)m_dataCPU.data(), m_dataGPU, sizeof(T) * dimX * dimY, cudaMemcpyDeviceToHost);
    }
    float& operator()(int x, int y)
    {
        return m_dataCPU[y * dimX + x];
    }
    float operator()(int x, int y) const
    {
        return m_dataCPU[y * dimX + x];
    }
    float& operator()(size_t x, size_t y)
    {
        return m_dataCPU[y * dimX + x];
    }
    float operator()(size_t x, size_t y) const
    {
        return m_dataCPU[y * dimX + x];
    }
    void clear(float clearValue)
    {
        for (int x = 0; x < dimX; x++)
            for (int y = 0; y < dimY; y++)
                (*this)(x, y) = clearValue;
    }
    static double maxDelta(const OptImage &a, const OptImage &b)
    {
        double delta = 0.0;
        for (int x = 0; x < a.dimX; x++)
            for (int y = 0; y < a.dimY; y++)
                delta = std::max(delta, fabs((double)a(x, y) - (double)b(x, y)));
        return delta;
    }
    static double avgDelta(const OptImage &a, const OptImage &b)
    {
        double delta = 0.0;
        for (int x = 0; x < a.dimX; x++)
            for (int y = 0; y < a.dimY; y++)
                delta += fabs((double)a(x, y) - (double)b(x, y));
        return delta / (double)(a.dimX * a.dimY);
    }
    const void * getDataCPU() const { return m_dataCPU.data(); }
    const void * getDataGPU() const { return m_dataGPU; }

    std::vector<T> m_dataCPU;
    T *m_dataGPU;
    int dimX, dimY;
};

typedef OptImage<float>		OptImagef;
typedef OptImage<float2>	OptImage2f;
typedef OptImage<float3>	OptImage3f;
typedef OptImage<float4>	OptImage4f;

#endif

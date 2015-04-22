#ifndef OptImage_h
#define OptImage_h
#include "cuda_runtime.h"

struct OptImage
{
    OptImage()
    {
        dataGPU = nullptr;
    }
    OptImage(int _dimX, int _dimY)
    {
        allocate(_dimX, _dimY);
    }

    void allocate(int _dimX, int _dimY)
    {
        dimX = _dimX;
        dimY = _dimY;
        dataCPU.resize(dimX * dimY);
        cudaMalloc(&dataGPU, sizeof(float) * dimX * dimY);
    }
    void syncCPUToGPU() const
    {
        cudaMemcpy(dataGPU, (void *)dataCPU.data(), sizeof(float) * dimX * dimY, cudaMemcpyHostToDevice);
    }
    void syncGPUToCPU() const
    {
        cudaMemcpy((void *)dataCPU.data(), dataGPU, sizeof(float) * dimX * dimY, cudaMemcpyDeviceToHost);
    }
    float& operator()(int x, int y)
    {
        return dataCPU[y * dimX + x];
    }
    float operator()(int x, int y) const
    {
        return dataCPU[y * dimX + x];
    }
    float& operator()(size_t x, size_t y)
    {
        return dataCPU[y * dimX + x];
    }
    float operator()(size_t x, size_t y) const
    {
        return dataCPU[y * dimX + x];
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
    const void * DataCPU() const { return dataCPU.data(); }
    const void * DataGPU() const { return dataGPU; }

    vector<float> dataCPU;
    void *dataGPU;
    int dimX, dimY;
};
#endif
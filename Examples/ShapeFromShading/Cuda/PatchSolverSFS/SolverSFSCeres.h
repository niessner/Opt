#pragma once

struct TerraSolverParameterPointers;


class CeresSolverSFS {

public:
    
    void solve(int width, int height, const std::vector<uint32_t>& elemsize, const std::vector<void*> &images, const TerraSolverParameterPointers &params);

private:
    int width, height;
};

#ifndef USE_CERES
inline void CeresSolverWarping::solve(float2* h_x_float, float* h_a_float, float2* h_urshape, float2* h_constraints, float* h_mask, float weightFit, float weightReg)
{

}

#endif
#pragma once

class CeresSolverPoissonImageEditing {

public:
    CeresSolverPoissonImageEditing(unsigned int _width, unsigned int _height)
	{
        width = _width;
        height = _height;
	}

    void solve(float4* h_unknown, float4* h_target, float* h_mask, float weightFit, float weightReg);

private:
    int width, height;
};

#ifndef USE_CERES

inline void CeresSolverPoissonImageEditing::solve(float4* h_unknown, float4* h_target, float* h_mask, float weightFit, float weightReg)
{

}

#endif
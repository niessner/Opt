#pragma once

class EigenSolverPoissonImageEditing {

public:
    EigenSolverPoissonImageEditing(unsigned int width, unsigned int height) : m_width(width), m_height(height) {}

    void solve(float4* h_unknown, float4* h_target, float* h_mask, float weightFit, float weightReg);

private:
    int m_width, m_height;
};

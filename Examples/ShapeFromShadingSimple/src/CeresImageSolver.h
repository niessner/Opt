#pragma once

#include <cassert>

#include "cutil.h"

class CeresImageSolver {

public:
    CeresImageSolver(unsigned int _width, unsigned int _height)
	{
        width = (int)_width;
        height = (int)_height;
	}

	struct IterStruct {
        unsigned int* nIter;
        unsigned int* lIter;
        unsigned int* pIter;
        IterStruct(unsigned int* n, unsigned int* l, unsigned int* p) : nIter(n), lIter(l), pIter(p) {}
    };

    void solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput);

    int getPixel(int x, int y) const
    {
        return y * width + x;
    }

    int width, height;
    float *Xfloat;
    float *D_i;
    float *Im;
    float *D_p;
    BYTE *edgeMaskR;
    BYTE *edgeMaskC;

    float w_p;
    float w_s;
    float w_r;
    float w_g;

    float weightShadingStart;
    float weightShadingIncrement;
    float weightBoundary;

    float f_x;
    float f_y;
    float u_x;
    float u_y;

    float4x4 deltaTransform;

    float L[9];
};

#ifndef USE_CERES

inline void CeresImageSolver::solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput)
{

}

#endif
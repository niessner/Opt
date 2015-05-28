#include "CuspSparseLaplacianSolverLinearOp.h"
#include <iostream>
#include "CUDATimer.h"

extern "C" void solve_linearOp(unsigned W, unsigned int H, float wFit, float wReg, float* d_image, float* d_target, int nNonLinearIterations, int nLinearIterations);

CuspSparseLaplacianSolverLinearOp::CuspSparseLaplacianSolverLinearOp(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
}

CuspSparseLaplacianSolverLinearOp::~CuspSparseLaplacianSolverLinearOp()
{
}

void CuspSparseLaplacianSolverLinearOp::solvePCG(float* d_image, float* d_target, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float weightFitting, float weightRegularizer)
{	
	int N  = m_imageWidth*m_imageHeight;
	solve_linearOp(m_imageWidth, m_imageHeight, weightFitting, weightRegularizer, d_image, d_target, nNonLinearIterations, nLinearIterations);
}

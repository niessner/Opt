#include "CuspSparseLaplacianSolver.h"
#include <iostream>
#include "CUDATimer.h"

extern "C" void solve(unsigned int N, unsigned int nz, float wFit, float wReg, float* target, float* x, int nIterations);

CuspSparseLaplacianSolver::CuspSparseLaplacianSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
}

CuspSparseLaplacianSolver::~CuspSparseLaplacianSolver()
{
}

void CuspSparseLaplacianSolver::solvePCG(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer)
{
	printf("Convergence of conjugate gradient without preconditioning: \n");
	
	CUDATimer timer;
	
	int N = m_imageWidth*m_imageHeight;
	int nz = 5 * N - 4 * (int)sqrt((double)N);

	float* h_x = new float[N];
	
	float* h_targetDepth = new float[N];
	cutilSafeCall(cudaMemcpy(h_targetDepth, d_targetDepth, N*sizeof(float), cudaMemcpyDeviceToHost)); // Free that array
	
	solve(N, nz, weightFitting, weightRegularizer, h_targetDepth, h_x, nIterations);
	
	cutilSafeCall(cudaMemcpy(d_result, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
}

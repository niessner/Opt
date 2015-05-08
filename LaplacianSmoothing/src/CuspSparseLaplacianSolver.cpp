#include "CuspSparseLaplacianSolver.h"
#include <iostream>
#include "CUDATimer.h"

void CuspSparseLaplacianSolver::genLaplace(int *row_ptr, int *col_ind, float *val, int N, int nz, float *rhs, float wFit, float wReg, float* target)
{
	int n = (int)sqrt((double)N);
	std::cout << n << std::endl;
	int idx = 0;
	for (int i = 0; i<N; i++)
	{
		int ix = i%n;
		int iy = i/n;

		row_ptr[i] = idx;

		int count = 0;
		if (iy > 0)		count++;
		if (ix > 0)		count++;
		if (ix < n - 1) count++;
		if (iy < n - 1) count++;

		// up
		if (iy > 0)
		{
			val[idx] = -1.0*wReg;
			col_ind[idx] = i - n;
			idx++;
		}

		// left
		if (ix > 0)
		{
			val[idx] = -1.0*wReg;
			col_ind[idx] = i - 1;
			idx++;
		}

		// center
		val[idx] = count*wReg+wFit;
		col_ind[idx] = i;
		idx++;

		rhs[i] = wFit*target[iy*n + ix];

		//right
		if (ix  < n - 1)
		{
			val[idx] = -1.0*wReg;
			col_ind[idx] = i + 1;
			idx++;
		}

		//down
		if (iy  < n - 1)
		{
			val[idx] = -1.0*wReg;
			col_ind[idx] = i + n;
			idx++;
		}
	}

	row_ptr[N] = idx;
}

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

	timer.startEvent("PCGInit");
	
	timer.endEvent();


		timer.startEvent("PCG");
	

		timer.endEvent();
		timer.nextIteration();
	
		timer.evaluate();

	//std::cout << "nIter: " << k << std::endl;
}

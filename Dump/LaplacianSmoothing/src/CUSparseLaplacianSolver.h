#pragma once

#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "LaplacianSolverState.h"

// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

class CUSparseLaplacianSolver
{
	public:
		CUSparseLaplacianSolver(unsigned int imageWidth, unsigned int imageHeight);
		~CUSparseLaplacianSolver();

		//! gauss newton
		void solvePCG(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer);
		
	private:

		void genLaplace(int *row_ptr, int *col_ind, float *val, int N, int nz, float *rhs, float wFit, float wReg, float* target);

		unsigned int m_imageWidth;
		unsigned int m_imageHeight;

		cublasHandle_t cublasHandle = 0;
		cublasStatus_t cublasStatus;

		cusparseHandle_t cusparseHandle = 0;
		cusparseStatus_t cusparseStatus;

		cusparseMatDescr_t descr = 0;

		int k, N = 0, nz = 0, *I = NULL, *J = NULL;
		int *d_col, *d_row;
		int qatest = 0;
		const float tol = 1e-12f;
		float *rhs;
		float r0, r1, alpha, beta;
		float *d_val;
		float *d_r, *d_p, *d_omega, *d_y;
		float *val = NULL;
		float rsum, diff, err = 0.0;
		float qaerr1, qaerr2 = 0.0;
		float dot, numerator, denominator, nalpha;
		const float floatone = 1.0;
		const float floatMinusOne = -1.0;
		const float floatzero = 0.0;

		int nErrors = 0;
};

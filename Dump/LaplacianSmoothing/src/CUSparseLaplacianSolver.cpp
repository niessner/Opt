#include "CUSparseLaplacianSolver.h"
#include <iostream>
#include "CUDATimer.h"

void CUSparseLaplacianSolver::genLaplace(int *row_ptr, int *col_ind, float *val, int N, int nz, float *rhs, float wFit, float wReg, float* target)
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

CUSparseLaplacianSolver::CUSparseLaplacianSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	N = m_imageWidth*m_imageHeight;
	nz = 5 * N - 4 * (int)sqrt((double)N);
	I = (int *)malloc(sizeof(int)*(N + 1));                            // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
	val = (float *)malloc(sizeof(float)*nz);                           // csr values for matrix A
	rhs = (float *)malloc(sizeof(float)*N);
}

CUSparseLaplacianSolver::~CUSparseLaplacianSolver()
{
	/* Destroy contexts */
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	/* Free device memory */
	free(I);
	free(J);
	free(val);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_y);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);
}

void CUSparseLaplacianSolver::solvePCG(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer)
{
	float* h_targetDepth = new float[N];
	cutilSafeCall(cudaMemcpy(h_targetDepth, d_targetDepth, N*sizeof(float), cudaMemcpyDeviceToHost)); // Free that array

	genLaplace(I, J, val, N, nz, rhs, 2*sqrtf(weightFitting), sqrtf(weightRegularizer), h_targetDepth);
		
	/* Create CUBLAS context */
	cublasStatus = cublasCreate(&cublasHandle);

	//checkCudaErrors(cublasStatus);

	/* Create CUSPARSE context */
	cusparseStatus = cusparseCreate(&cusparseHandle);

	//checkCudaErrors(cusparseStatus);

	/* Description of the A matrix*/
	cusparseStatus = cusparseCreateMatDescr(&descr);

	//checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	/* Allocate required memory */
	cutilSafeCall(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_row, (N + 1)*sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_y, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_r, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_p, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_omega, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

	/* Conjugate gradient without preconditioning.
	------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

	printf("Convergence of conjugate gradient without preconditioning: \n");
	
	CUDATimer timer;

	timer.startEvent("PCGInit");
		k = 0;
		r0 = 0;
		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatMinusOne, descr, d_val, d_row, d_col, d_result, &floatone, d_r);
		cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	timer.endEvent();

	while (k < nIterations) // r1 > tol*tol && 
	{
		timer.startEvent("PCG");
		k++;

		if (k == 1)
		{
			cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
		}
		else
		{
			beta = r1 / r0;
			cublasSscal(cublasHandle, N, &beta, d_p, 1);
			cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1);
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);
		cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
		alpha = r1 / dot;
		cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_result, 1);
		nalpha = -alpha;
		cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
		r0 = r1;
		cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

		timer.endEvent();
		timer.nextIteration();
	}
	timer.evaluate();

	std::cout << "nIter: " << k << std::endl;
}

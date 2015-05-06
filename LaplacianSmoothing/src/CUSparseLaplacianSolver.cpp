#include "CUSparseLaplacianSolver.h"

void CUSparseLaplacianSolver::genLaplace(int *row_ptr, int *col_ind, float *val, int M, int N, int nz, float *rhs)
{
	//assert(M == N);
	int n = (int)sqrt((double)N);
	//assert(n*n == N);
	printf("laplace dimension = %d\n", n);
	int idx = 0;

	// loop over degrees of freedom
	for (int i = 0; i<N; i++)
	{
		int ix = i%n;
		int iy = i / n;

		row_ptr[i] = idx;

		// up
		if (iy > 0)
		{
			val[idx] = 1.0;
			col_ind[idx] = i - n;
			idx++;
		}
		else
		{
			rhs[i] -= 1.0;
		}

		// left
		if (ix > 0)
		{
			val[idx] = 1.0;
			col_ind[idx] = i - 1;
			idx++;
		}
		else
		{
			rhs[i] -= 0.0;
		}

		// center
		val[idx] = -4.0;
		col_ind[idx] = i;
		idx++;

		//right
		if (ix  < n - 1)
		{
			val[idx] = 1.0;
			col_ind[idx] = i + 1;
			idx++;
		}
		else
		{
			rhs[i] -= 0.0;
		}

		//down
		if (iy  < n - 1)
		{
			val[idx] = 1.0;
			col_ind[idx] = i + n;
			idx++;
		}
		else
		{
			rhs[i] -= 0.0;
		}

	}

	row_ptr[N] = idx;

}

CUSparseLaplacianSolver::CUSparseLaplacianSolver(unsigned int imageWidth, unsigned int imageHeight) : m_imageWidth(imageWidth), m_imageHeight(imageHeight)
{
	//N = imageWidth*imageHeight;

	/* Generate a random tridiagonal symmetric matrix in CSR (Compressed Sparse Row) format */
	M = N = 16384;
	nz = 5 * N - 4 * (int)sqrt((double)N);
	I = (int *)malloc(sizeof(int)*(N + 1));                            // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
	val = (float *)malloc(sizeof(float)*nz);                           // csr values for matrix A
	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = 0.0;                                                  // Initialize RHS
		x[i] = 0.0;                                                    // Initial approximation of solution
	}

	genLaplace(I, J, val, M, N, nz, rhs);

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
	cutilSafeCall(cudaMalloc((void **)&d_x, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_y, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_r, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_p, N*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_omega, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);
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
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);
}

void CUSparseLaplacianSolver::solvePCG(float* d_targetDepth, float* d_result, unsigned int nIterations, float weightFitting, float weightRegularizer)
{
	/* Conjugate gradient without preconditioning.
	------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

	printf("Convergence of conjugate gradient without preconditioning: \n");
	k = 0;
	r0 = 0;
	cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

	while (r1 > tol*tol && k <= nIterations)
	{
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
		cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
		nalpha = -alpha;
		cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
		r0 = r1;
		cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	}

	printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	/* check result */
	err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	printf("  Convergence Test: %s \n", (k <= nIterations) ? "OK" : "FAIL");
	nErrors += (k > nIterations) ? 1 : 0;
	qaerr1 = err;

	if (0)
	{
		// output result in matlab-style array
		int n = (int)sqrt((double)N);
		printf("a = [  ");

		for (int iy = 0; iy<n; iy++)
		{
			for (int ix = 0; ix<n; ix++)
			{
				printf(" %f ", x[iy*n + ix]);
			}

			if (iy == n - 1)
			{
				printf(" ]");
			}

			printf("\n");
		}
	}

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits

	printf("  Test Summary:\n");
	printf("     Counted total of %d errors\n", nErrors);
	printf("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
}

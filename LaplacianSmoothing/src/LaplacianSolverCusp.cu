#include "cusp/hyb_matrix.h"
#include "cusp/csr_matrix.h"
#include "cusp/gallery/poisson.h"
#include "cusp/krylov/cg.h"
#include <cusp/precond/diagonal.h>

void genLaplace(cusp::csr_matrix<int, float, cusp::host_memory>* A, int N, int nz, cusp::array1d<float, cusp::host_memory>* rhs, float wFit, float wReg, float* target)
{
	int n = (int)sqrt((double)N);
	std::cout << n << std::endl;
	int idx = 0;
	for (int i = 0; i<N; i++)
	{
		int ix = i%n;
		int iy = i / n;

		A->row_offsets[i] = idx;

		int count = 0;
		if (iy > 0)		count++;
		if (ix > 0)		count++;
		if (ix < n - 1) count++;
		if (iy < n - 1) count++;

		// up
		if (iy > 0)
		{
			A->values[idx] = -1.0*wReg;
			A->column_indices[idx] = i - n;
			idx++;
		}

		// left
		if (ix > 0)
		{
			A->values[idx] = -1.0*wReg;
			A->column_indices[idx] = i - 1;
			idx++;
		}

		// center
		A->values[idx] = count*wReg + wFit;
		A->column_indices[idx] = i;
		idx++;

		(*rhs)[i] = wFit*target[iy*n + ix];

		//right
		if (ix  < n - 1)
		{
			A->values[idx] = -1.0*wReg;
			A->column_indices[idx] = i + 1;
			idx++;
		}

		//down
		if (iy  < n - 1)
		{
			A->values[idx] = -1.0*wReg;
			A->column_indices[idx] = i + n;
			idx++;
		}
	}

	A->row_offsets[N] = idx;
}

int solve(unsigned int N, unsigned int nz, float wFit, float wReg, float* target)
{
	cusp::csr_matrix<int, float, cusp::host_memory> A_CPU(N, N, nz);
	cusp::array1d<float, cusp::host_memory>			b_CPU(N, 0);

	genLaplace(&A_CPU, N, nz, &b_CPU, wFit, wReg, target);

	A_CPU.row_offsets[0];
	A_CPU.column_indices[0]; A_CPU.values[0] = 10;

	cusp::csr_matrix<int, float, cusp::device_memory>	A_GPU(A_CPU);
	cusp::precond::diagonal<float, cusp::device_memory> M_GPU(A_GPU);
	cusp::array1d<float, cusp::device_memory>			b_GPU(b_CPU);
	cusp::array1d<float, cusp::device_memory>			x_GPU(N, 0);

	cusp::verbose_monitor<float> monitor(b_GPU,100, 0, 0);
	cusp::krylov::cg(A_GPU, x_GPU, b_GPU, monitor, M_GPU);

	return 0;
}

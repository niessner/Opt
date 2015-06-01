#include "cusp/hyb_matrix.h"
#include "cusp/csr_matrix.h"
#include "cusp/gallery/poisson.h"
#include "cusp/krylov/cg.h"
#include <cusp/precond/diagonal.h>
#include <cusp/linear_operator.h>

#include "CUDATimer.h"

void genLaplace2(cusp::csr_matrix<int, float, cusp::host_memory>* A, int N, int nz, cusp::array1d<float, cusp::host_memory>* rhs, float wFit, float wReg, float* target)
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

__global__ void stencil_kernel(int N, const float * x, float * y, float wFit, float wReg)
{
	//__shared__ float shared[1 + 16 + 1][1 + 16 + 1];

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//int linearizedIndex = 16*threadIdx.x + threadIdx.y;
	//for (int k = linearizedIndex; k<(1+16+1)*(1+16+1); k+=16*16)
	//{
	//	int jMem = k%(1+16+1)-1;
	//	int iMem = k/(1+16+1)-1;
	//
	//	int jGlobal = blockDim.y * blockIdx.y + jMem;
	//	int iGlobal = blockDim.x * blockIdx.x + iMem;
	//
	//	if (jGlobal >= 0 && jGlobal < N && iGlobal >= 0 && iGlobal < N) shared[jMem + 1][iMem + 1] = x[N*iGlobal + jGlobal];
	//	else															shared[jMem + 1][iMem + 1] = -1000;
	//}
	//
	//__syncthreads();
	//
	//if (i < N && j < N)
	//{
	//	int index = N * i + j;
	//
	//	float count  = 0.0f;
	//	float result = 0.0f;
	//	float val = shared[(threadIdx.y + 1) - 1][(threadIdx.x + 1)	   ]; if (val != -1000) { result -= wReg*val;  count++; }
	//		  val = shared[(threadIdx.y + 1)	][(threadIdx.x + 1) - 1]; if (val != -1000) { result -= wReg*val;  count++; }
	//		  val = shared[(threadIdx.y + 1)	][(threadIdx.x + 1) + 1]; if (val != -1000) { result -= wReg*val;  count++; }
	//		  val = shared[(threadIdx.y + 1) + 1][(threadIdx.x + 1)	   ]; if (val != -1000) { result -= wReg*val;  count++; }
	//	
	//	result += (count*wReg + wFit)*shared[(threadIdx.y + 1)][(threadIdx.x + 1)];
	//	
	//	y[N*i + j] = result;
	//}

	if (i < N && j < N)
	{
		int index = N * i + j;
	
		float count  = 0.0f;
		float result = 0.0f;
		if (i > 0)		{ result -= wReg*x[index - N];  count++; }
		if (j > 0)		{ result -= wReg*x[index - 1];  count++; }
		if (j < N - 1)  { result -= wReg*x[index + 1];  count++; }
		if (i < N - 1)	{ result -= wReg*x[index + N];  count++; }
		result += (count*wReg + wFit)*x[index];
		
		y[N*i + j] = result;
	}
}

class stencil : public cusp::linear_operator<float, cusp::device_memory>
{
public:
	typedef cusp::linear_operator<float, cusp::device_memory> super;

	int N;
	float wFit;
	float wReg;

	// constructor
	stencil(int N, float wFit, float wReg)
		: super(N*N, N*N), N(N), wFit(wFit), wReg(wReg) 
	{}

	// linear operator y = A*x
	template <typename VectorType1,
		typename VectorType2>
		void operator()(const VectorType1& x, VectorType2& y) const
	{
			// obtain a raw pointer to device memory
			const float * x_ptr = thrust::raw_pointer_cast(&x[0]);
			float * y_ptr = thrust::raw_pointer_cast(&y[0]);

			dim3 dimBlock(16, 16);
			dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

			stencil_kernel << <dimGrid, dimBlock >> >(N, x_ptr, y_ptr, wFit, wReg);
	}
};

extern "C" void solve_linearOp(unsigned int N, unsigned int nz, float wFit, float wReg, float* target, float* x, int nIterations)
{
	cusp::csr_matrix<int, float, cusp::host_memory> A_CPU(N, N, nz);
	cusp::array1d<float, cusp::host_memory>			b_CPU(N, 0);

	genLaplace2(&A_CPU, N, nz, &b_CPU, 2*sqrtf(wFit), sqrtf(wReg), target);

	stencil A(sqrt(N), 2*sqrtf(wFit), sqrtf(wReg));
	
	cusp::csr_matrix<int, float, cusp::device_memory>	A_GPU(A_CPU);
	cusp::precond::diagonal<float, cusp::device_memory> M_GPU(A_GPU);
	cusp::array1d<float, cusp::device_memory>			b_GPU(b_CPU);
	cusp::array1d<float, cusp::device_memory>			x_GPU(N, 0);

	cusp::default_monitor<float> monitor(b_GPU, nIterations, 0, 0);
	
	CUDATimer timer;
	timer.startEvent("PCG");

	cusp::krylov::cg(A, x_GPU, b_GPU, monitor, M_GPU);

	timer.endEvent();
	timer.nextIteration();

	cusp::array1d<float, cusp::host_memory> x_CPU(x_GPU);
	for(unsigned int i = 0; i < N; i++) x[i] = x_CPU[i];

	timer.evaluate();
}

#include "cusp/hyb_matrix.h"
#include "cusp/csr_matrix.h"
#include "cusp/gallery/poisson.h"
#include "cusp/krylov/cg.h"
#include <cusp/precond/diagonal.h>
#include <cusp/linear_operator.h>
#include <cuda_runtime.h> 
#include <cutil_inline.h>

#include "CUDATimer.h"

void genPrecond(cusp::csr_matrix<int, float, cusp::host_memory>* M, int W, int H, int N, cusp::array1d<float, cusp::host_memory>* rhs, float wFit, float wReg, float* target)
{
	int idx = 0;
	for (int i = 0; i<N; i++)
	{
		int ix = i%W;
		int iy = i/W;

		M->row_offsets[i] = idx;

		int count = 0;
		if (iy > 0)		count++;
		if (ix > 0)		count++;
		if (ix < W - 1) count++;
		if (iy < H - 1) count++;

		// center
		M->values[idx] = count*wReg + wFit;
		M->column_indices[idx] = i;
		idx++;

		(*rhs)[i] = wFit*target[iy*W + ix];
	}

	M->row_offsets[N] = idx;
}

__global__ void stencil_kernel(int W, int H, const float * x, float * y, float wFit, float wReg)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < H && j < W)
	{
		int index = W * i + j;
	
		float count  = 0.0f;
		float result = 0.0f;
		if (i > 0)		{ result -= wReg*x[index - W];  count++; }
		if (j > 0)		{ result -= wReg*x[index - 1];  count++; }
		if (j < W - 1)  { result -= wReg*x[index + 1];  count++; }
		if (i < H - 1)	{ result -= wReg*x[index + W];  count++; }
		result += (count*wReg + wFit)*x[index];
		
		y[W*i + j] = result;
	}
}

class stencil : public cusp::linear_operator<float, cusp::device_memory>
{
	public:

		typedef cusp::linear_operator<float, cusp::device_memory> super;

		int W; int H;
		float wFit;	float wReg;

		stencil(int W, int H, float wFit, float wReg) : super(W*H, W*H), W(W), H(H), wFit(wFit), wReg(wReg) 
		{
		}

		template <typename VectorType1, typename VectorType2>
		void operator()(const VectorType1& x, VectorType2& y) const
		{
				// obtain a raw pointer to device memory
				const float* x_ptr = thrust::raw_pointer_cast(&x[0]);
				float*		 y_ptr = thrust::raw_pointer_cast(&y[0]);

				const unsigned int dim = 16;
				dim3 dimBlock(dim, dim);
				dim3 dimGrid((W + dim - 1)/dim, (H + dim - 1)/dim);

				stencil_kernel << <dimGrid, dimBlock >> >(W, H, x_ptr, y_ptr, wFit, wReg);
		}
};

extern "C" void solve_linearOp(unsigned W, unsigned int H, float wFit, float wReg, float* d_image, float* d_target, int nNonLinearIterations, int nLinearIterations)
{
	unsigned int N = W*H;
	unsigned int nz = N;

	cusp::csr_matrix<int, float, cusp::host_memory> M_CPU(N, N, nz);
	cusp::array1d<float, cusp::host_memory>			b_CPU(N, 0);
	cusp::array1d<float, cusp::host_memory>			x_CPU(N, 0);
	
	cutilSafeCall(cudaMemcpy(&x_CPU[0], d_image, N*sizeof(float), cudaMemcpyDeviceToHost));
	float* target = new float[N]; cutilSafeCall(cudaMemcpy(target, d_target, N*sizeof(float), cudaMemcpyDeviceToHost));
	genPrecond(&M_CPU, W, H, N, &b_CPU, 2 * sqrtf(wFit), sqrtf(wReg), target);
	delete target;

	stencil A(W, H, 2*sqrtf(wFit), sqrtf(wReg));

	cusp::csr_matrix<int, float, cusp::device_memory> M_GPU(M_CPU);
	cusp::precond::diagonal<float, cusp::device_memory> M(M_GPU);
	cusp::array1d<float, cusp::device_memory> b_GPU(b_CPU);
	cusp::array1d<float, cusp::device_memory> x_GPU(x_CPU);
		
	cusp::default_monitor<float> monitor(b_GPU, nLinearIterations, 0, 0);
	
	CUDATimer timer;
	for (unsigned int nIter = 0; nIter < nNonLinearIterations; nIter++)
	{
		timer.startEvent("PCG");
			cusp::krylov::cg(A, x_GPU, b_GPU, monitor, M);
		timer.endEvent();
		timer.nextIteration();
	}
	timer.evaluate();
	

	cusp::array1d<float, cusp::host_memory>	res(x_GPU); // not ideal!
	cutilSafeCall(cudaMemcpy(d_image, &res[0], N*sizeof(float), cudaMemcpyHostToDevice));
}

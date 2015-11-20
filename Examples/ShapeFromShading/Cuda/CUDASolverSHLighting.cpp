#include "stdafx.h"
#include "CUDASolverSHLighting.h"
#include "../Eigen.h"
#include "GlobalAppState.h"

extern "C" void estimateLightingSH(SolverSHInput& input, float4* d_normalMap, float *d_litestcashe, float *h_litestmat, float thres_depth);
extern "C" void estimateReflectance(SolverSHInput& input, float4* d_normalMap);

CUDASolverSHLighting::CUDASolverSHLighting(int width, int height)
{
	const unsigned int numoflecache = (width+LE_THREAD_SIZE-1)/LE_THREAD_SIZE*height*54;
	cutilSafeCall(cudaMalloc(&d_litestcashe, sizeof(float)*3*numoflecache));
	h_litestmat = new float[54];
}

CUDASolverSHLighting::~CUDASolverSHLighting()
{
	cutilSafeCall(cudaFree(d_litestcashe));	
	delete [] h_litestmat;

}

void CUDASolverSHLighting::solveLighting(SolverSHInput &input, float4* d_normalMap, float thres_depth)
{	
	estimateLightingSH(input, d_normalMap, d_litestcashe, h_litestmat, thres_depth);
	
	Eigen::MatrixXf A(9,9);
	Eigen::VectorXf b(9);
	int move_pt = 0;
	for(unsigned int i=0;i<9;i++)
	{
		for(unsigned int j=i;j<9;j++)
		{
			A(i,j) = h_litestmat[move_pt];
			A(j,i) = h_litestmat[move_pt];
			move_pt++;
		}
		b(i) = h_litestmat[45+i];
	} 

	if(input.d_litprior != NULL)
	{
		cutilSafeCall(cudaMemcpy(h_litestmat, input.d_litprior, 9*sizeof(float), cudaMemcpyDeviceToHost));
		float weight_prior = GlobalAppState::get().s_weightPriorLight;

		for(unsigned int i=0;i<9;i++)
		{
			A(i,i) += weight_prior;
			b(i) += h_litestmat[i]*weight_prior;
		}
	}
	
	Eigen::VectorXf x = A.llt().solve(b);

	
	for(unsigned int i=0;i<9;i++)
		h_litestmat[i] = x(i);	
	

	cutilSafeCall(cudaMemcpy(input.d_litcoeff,h_litestmat,9*sizeof(float),cudaMemcpyHostToDevice));
}



void CUDASolverSHLighting::solveReflectance(SolverSHInput &input, float4* d_normalMap)
{
	estimateReflectance(input, d_normalMap);
}

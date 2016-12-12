#pragma once

#define RUN_CUDA 0
#define RUN_TERRA 0
#define RUN_OPT 1

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraWarpingSolver.h"
#include "CUDAWarpingSolver.h"
#include "OpenMesh.h"

class ImageWarping
{
public:
	ImageWarping(const SimpleMesh* mesh)
	{
		m_result = *mesh;
		m_initial = m_result;

		unsigned int N = (unsigned int)mesh->n_vertices();
		unsigned int E = (unsigned int)mesh->n_edges();

		cutilSafeCall(cudaMalloc(&d_vertexPosTargetFloat3, sizeof(float3)*N));

		cutilSafeCall(cudaMalloc(&d_vertexPosFloat3, sizeof(float3)*N));
		cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
		cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int) * 2 * E));
		cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N + 1)));

		resetGPUMemory();
		m_warpingSolver = new CUDAWarpingSolver(N);
#if RUN_TERRA
		m_terraWarpingSolver = new TerraWarpingSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshSmoothingLaplacian.t", "gaussNewtonGPU");
#endif
		m_optWarpingSolver = new TerraWarpingSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshSmoothingLaplacianAD.t", "gaussNewtonGPU");
	}

	void resetGPUMemory()
	{
		unsigned int N = (unsigned int)m_initial.n_vertices();
		unsigned int E = (unsigned int)m_initial.n_edges();
		float3* h_vertexPosFloat3 = new float3[N];
		int*	h_numNeighbours = new int[N];
		int*	h_neighbourIdx = new int[2 * E];
		int*	h_neighbourOffset = new int[N + 1];

		for (unsigned int i = 0; i < N; i++)
		{
			const Vec3f& pt = m_initial.point(VertexHandle(i));
			h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
		}

		unsigned int count = 0;
		unsigned int offset = 0;
		h_neighbourOffset[0] = 0;
		for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
		{
			VertexHandle c_vh(v_it.handle());
			//				printf("%d\n", c_vh.idx());
			unsigned int valance = m_initial.valence(c_vh);
			h_numNeighbours[count] = valance;

			for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++)
			{
				VertexHandle v_vh(vv_it.handle());

				h_neighbourIdx[offset] = v_vh.idx();
				offset++;
			}

			h_neighbourOffset[count + 1] = offset;

			count++;
		}

		cutilSafeCall(cudaMemcpy(d_vertexPosTargetFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, sizeof(int)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, sizeof(int) * 2 * E, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, sizeof(int)*(N + 1), cudaMemcpyHostToDevice));

		delete[] h_vertexPosFloat3;
		delete[] h_numNeighbours;
		delete[] h_neighbourIdx;
		delete[] h_neighbourOffset;
	}

	~ImageWarping()
	{
		cutilSafeCall(cudaFree(d_vertexPosTargetFloat3));
		cutilSafeCall(cudaFree(d_vertexPosFloat3));
		cutilSafeCall(cudaFree(d_numNeighbours));
		cutilSafeCall(cudaFree(d_neighbourIdx));
		cutilSafeCall(cudaFree(d_neighbourOffset));

		SAFE_DELETE(m_warpingSolver);
		SAFE_DELETE(m_terraWarpingSolver);
		SAFE_DELETE(m_optWarpingSolver);
	}

	SimpleMesh* solve()
	{
		float weightFit = 50.0f;
		float weightReg = 100.0f;

		unsigned int nonLinearIter = 10;
		unsigned int linearIter = 10;

#		if RUN_CUDA
		std::cout << "=========CUDA========" << std::endl;
		resetGPUMemory();
		m_warpingSolver->solveGN(d_vertexPosFloat3, d_numNeighbours, d_neighbourIdx, d_neighbourOffset, d_vertexPosTargetFloat3, nonLinearIter, linearIter, weightFit, weightReg);
		copyResultToCPUFromFloat3();
#			endif

#		if RUN_TERRA
		std::cout << "=========TERRA=======" << std::endl;
		resetGPUMemory();
		m_terraWarpingSolver->solve(d_vertexPosFloat3, d_vertexPosTargetFloat3, nonLinearIter, linearIter, 1, weightFit, weightReg);
		copyResultToCPUFromFloat3();
#		endif

#		if RUN_OPT
		std::cout << "=========OPT=========" << std::endl;
		resetGPUMemory();
		m_optWarpingSolver->solve(d_vertexPosFloat3, d_vertexPosTargetFloat3, nonLinearIter, linearIter, 1, weightFit, weightReg);
		copyResultToCPUFromFloat3();
#		endif


		return &m_result;
	}

	void copyResultToCPUFromFloat3()
	{
		unsigned int N = (unsigned int)m_result.n_vertices();
		float3* h_vertexPosFloat3 = new float3[N];
		cutilSafeCall(cudaMemcpy(h_vertexPosFloat3, d_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyDeviceToHost));

		for (unsigned int i = 0; i < N; i++)
		{
			m_result.set_point(VertexHandle(i), Vec3f(h_vertexPosFloat3[i].x, h_vertexPosFloat3[i].y, h_vertexPosFloat3[i].z));
		}

		delete[] h_vertexPosFloat3;
	}

private:

	SimpleMesh m_result;
	SimpleMesh m_initial;

	float3*	d_vertexPosTargetFloat3;
	float3*	d_vertexPosFloat3;
	int*	d_numNeighbours;
	int*	d_neighbourIdx;
	int* 	d_neighbourOffset;

	TerraWarpingSolver* m_optWarpingSolver;
	TerraWarpingSolver* m_terraWarpingSolver;
	CUDAWarpingSolver* m_warpingSolver;
};

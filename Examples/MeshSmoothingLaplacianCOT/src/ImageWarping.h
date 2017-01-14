#pragma once

#include "../../shared/Precision.h"
#include "../../shared/OptUtils.h"
#include "../../shared/CombinedSolverParameters.h"
#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"

struct EdgeCOT {
	unsigned int v0;	//current
	unsigned int v1;	//neighbor
	unsigned int v2;	//prev neigh
	unsigned int v3;	//next neigh
};


class ImageWarping
{
public:
	ImageWarping(const SimpleMesh* mesh, bool performanceRun)
	{
		m_result = *mesh;
		m_initial = m_result;

		unsigned int N = (unsigned int)mesh->n_vertices();
		unsigned int E = (unsigned int)mesh->n_edges();

		cutilSafeCall(cudaMalloc(&d_vertexPosTargetFloat3, sizeof(float3)*N));

		cutilSafeCall(cudaMalloc(&d_vertexPosFloat3, sizeof(float3)*N));
		cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
		cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int) * 2 * E * 3));
		cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N + 1)));
		//m_params.useOpt = false;
		//m_params.useOptLM = true;//performanceRun;
        m_params.nonLinearIter = 8;
        m_params.linearIter = 25;


		resetGPUMemory();
		std::cout << "compiling... ";
		m_optWarpingSolver = std::unique_ptr<TerraWarpingSolver>(new TerraWarpingSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshSmoothingLaplacianAD.t", "gaussNewtonGPU"));
        m_optLMWarpingSolver = std::unique_ptr<TerraWarpingSolver>(new TerraWarpingSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshSmoothingLaplacianAD.t", "LMGPU"));
		std::cout << " done!" << std::endl;
	}

	void resetGPUMemory()
	{
		unsigned int N = (unsigned int)m_initial.n_vertices();
		unsigned int E = (unsigned int)m_initial.n_edges();
		float3* h_vertexPosFloat3 = new float3[N];
		int*	h_numNeighbours = new int[N];
		int*	h_neighbourIdx = new int[2 * E * 3];
		int*	h_neighbourOffset = new int[N + 1];

		bool isTri = m_initial.is_triangles();
		if (!isTri) {
			std::cout << "MUST BE A TRI MESH" << std::endl;
			exit(1);
		}
		
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
			unsigned int valance = m_initial.valence(c_vh);
			h_numNeighbours[count] = valance;


			//for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++) {
			//	VertexHandle v_vh(vv_it.handle());
			//	h_neighbourIdx[offset] = v_vh.idx();
			//	offset++;
			//}

			std::vector<unsigned int> neighborIndices;
			for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++) {
				VertexHandle v_vh(vv_it.handle());
				neighborIndices.push_back(v_vh.idx());
				//h_neighbourIdx[offset] = v_vh.idx();
				//offset++;
			}

			for (size_t i = 0; i < neighborIndices.size(); i++) {
				const unsigned int n = (unsigned int)neighborIndices.size();
				unsigned int prev = neighborIndices[(i + n - 1) % n];
				unsigned int next = neighborIndices[(i + 1) % n];
				unsigned int curr = neighborIndices[i];

				h_neighbourIdx[3*offset+0] = curr;
				h_neighbourIdx[3*offset+1] = prev;
				h_neighbourIdx[3*offset+2] = next;
				offset++;
			}

			h_neighbourOffset[count + 1] = offset;

			count++;
		}

		cutilSafeCall(cudaMemcpy(d_vertexPosTargetFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, sizeof(int)*N, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, sizeof(int) * 2 * E * 3, cudaMemcpyHostToDevice));
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
	}

	SimpleMesh* solve()
	{
		float weightFit = 1.0f;
		float weightReg = 4.5f;


        std::vector<SolverIteration> ceresIters, optIters, optLMIters;

        if (m_params.useOpt) {
            std::cout << "=========OPT=========" << std::endl;
            resetGPUMemory();
            unsigned int numIter = 1;
            for (unsigned int i = 0; i < numIter; i++) {
                m_optWarpingSolver->solve(d_vertexPosFloat3, d_vertexPosTargetFloat3, m_params.nonLinearIter, m_params.linearIter, 1, weightFit, weightReg, optIters);
            }
            copyResultToCPUFromFloat3();
        }

        if (m_params.useOptLM) {
            std::cout << "=========OPT LM=========" << std::endl;
            resetGPUMemory();
            unsigned int numIter = 1;
            for (unsigned int i = 0; i < numIter; i++) {
                m_optLMWarpingSolver->solve(d_vertexPosFloat3, d_vertexPosTargetFloat3, m_params.nonLinearIter, m_params.linearIter, 1, weightFit, weightReg, optLMIters);
            }
            copyResultToCPUFromFloat3();
        }

        
        std::string resultDirectory = "results/";
#   if OPT_DOUBLE_PRECISION
        std::string resultSuffix = "_double";
#   else
        std::string resultSuffix = "_float";
#   endif
        saveSolverResults(resultDirectory, resultSuffix, ceresIters, optIters, optLMIters);

        reportFinalCosts("Cotangent Mesh Laplacian", m_params, m_optWarpingSolver->finalCost(), m_optLMWarpingSolver->finalCost(), nan(nullptr));

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

    CombinedSolverParameters m_params;

	std::unique_ptr<TerraWarpingSolver> m_optWarpingSolver;
    std::unique_ptr<TerraWarpingSolver> m_optLMWarpingSolver;
};

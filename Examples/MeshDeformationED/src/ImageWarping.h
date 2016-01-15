#pragma once

#define RUN_OPT 1

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraSolver.h"
#include "OpenMesh.h"

class ImageWarping
{
	public:
		ImageWarping(const SimpleMesh* mesh, std::vector<int> constraintsIdx, std::vector<std::vector<float>> constraintsTarget) : m_constraintsIdx(constraintsIdx), m_constraintsTarget(constraintsTarget)
		{
			m_result = *mesh;
			m_initial = m_result;

			unsigned int N = (unsigned int)mesh->n_vertices();
			unsigned int E = (unsigned int)mesh->n_edges();

			cutilSafeCall(cudaMalloc(&d_vertexPosTargetFloat3, sizeof(float3)*N));

			cutilSafeCall(cudaMalloc(&d_vertexPosFloat3, sizeof(float3)*N));
			cutilSafeCall(cudaMalloc(&d_vertexPosFloat3Urshape, sizeof(float3)*N));
			cutilSafeCall(cudaMalloc(&d_rotsFloat9, sizeof(float9)*N));
			cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
			cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
			cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));
		
			resetGPUMemory();	
			m_optSolver = new TerraSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshDeformationEDSimple.t", "gaussNewtonGPU");
		} 

		void setConstraints(float alpha)
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
			float3* h_vertexPosTargetFloat3 = new float3[N];
			for (unsigned int i = 0; i < N; i++)
			{
				h_vertexPosTargetFloat3[i] = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
			}

			for (unsigned int i = 0; i < m_constraintsIdx.size(); i++)
			{
				const Vec3f& pt = m_result.point(VertexHandle(m_constraintsIdx[i]));
				const Vec3f target = Vec3f(m_constraintsTarget[i][0], m_constraintsTarget[i][1], m_constraintsTarget[i][2]);

				Vec3f z = (1 - alpha)*pt + alpha*target;
				h_vertexPosTargetFloat3[m_constraintsIdx[i]] = make_float3(z[0], z[1], z[2]);
			}
			cutilSafeCall(cudaMemcpy(d_vertexPosTargetFloat3, h_vertexPosTargetFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
			delete [] h_vertexPosTargetFloat3;
		}

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();
			unsigned int E = (unsigned int)m_initial.n_edges();

			float3* h_vertexPosFloat3 = new float3[N];
			int*	h_numNeighbours   = new int[N];
			int*	h_neighbourIdx	  = new int[2*E];
			int*	h_neighbourOffset = new int[N+1];

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

				for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++)
				{
					VertexHandle v_vh(vv_it.handle());

					h_neighbourIdx[offset] = v_vh.idx();
					offset++;
				}

				h_neighbourOffset[count + 1] = offset;

				count++;
			}
			
			// Constraints
			setConstraints(1.0f);


			// Angles
			ml::mat3f* h_rots = new ml::mat3f[N];
			for (unsigned int i = 0; i < N; i++)
			{
				h_rots[i].setIdentity();
			}
			cutilSafeCall(cudaMemcpy(d_rotsFloat9, h_rots, sizeof(float3)*N, cudaMemcpyHostToDevice));
			delete [] h_rots;
			
			cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_vertexPosFloat3Urshape, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, sizeof(int)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, sizeof(int)* 2 * E, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, sizeof(int)*(N + 1), cudaMemcpyHostToDevice));

			delete [] h_vertexPosFloat3;
			delete [] h_numNeighbours;
			delete [] h_neighbourIdx;
			delete [] h_neighbourOffset;
		}

		~ImageWarping()
		{
			cutilSafeCall(cudaFree(d_rotsFloat9));

			cutilSafeCall(cudaFree(d_vertexPosTargetFloat3));
			cutilSafeCall(cudaFree(d_vertexPosFloat3));
			cutilSafeCall(cudaFree(d_vertexPosFloat3Urshape));
			cutilSafeCall(cudaFree(d_numNeighbours));
			cutilSafeCall(cudaFree(d_neighbourIdx));
			cutilSafeCall(cudaFree(d_neighbourOffset));

			SAFE_DELETE(m_optSolver);
		}

		SimpleMesh* solve()
		{
			
			//float weightFit = 5.0f;
			float weightFit = 3.0f;
			float weightReg = 4.0f;
			float weightRot = 5.0f;
		
            unsigned int numIter = 32;
            unsigned int nonLinearIter = 1;
            unsigned int linearIter = 4000;

			//unsigned int numIter = 2;
			//unsigned int nonLinearIter = 3;
			//unsigned int linearIter = 3;

#			if RUN_OPT
			m_result = m_initial;
			resetGPUMemory();
			for (unsigned int i = 1; i < numIter; i++)
			{
				std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
				setConstraints((float)i / (float)(numIter - 1));

				m_optSolver->solveGN(d_vertexPosFloat3, d_rotsFloat9, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, nonLinearIter, linearIter, weightFit, weightReg, weightRot);
			}
			copyResultToCPUFromFloat3();
#			endif
						
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

			delete [] h_vertexPosFloat3;
		}

	private:

		SimpleMesh m_result;
		SimpleMesh m_initial;
	
		float9* d_rotsFloat9;
		float3*	d_vertexPosTargetFloat3;
		float3*	d_vertexPosFloat3;
		float3*	d_vertexPosFloat3Urshape;
		int*	d_numNeighbours;
		int*	d_neighbourIdx;
		int* 	d_neighbourOffset;

		TerraSolver* m_optSolver;

		std::vector<int>				m_constraintsIdx;
		std::vector<std::vector<float>>	m_constraintsTarget;
};

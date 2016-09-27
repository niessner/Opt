#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"

class ImageWarping
{
	public:
        ImageWarping(const SimpleMesh* sourceMesh, const SimpleMesh* targetMesh)
		{
            m_result = *sourceMesh;
			m_initial = m_result;
            m_target = *targetMesh;

            unsigned int N = (unsigned int)sourceMesh->n_vertices();
            unsigned int E = (unsigned int)sourceMesh->n_edges();

			cutilSafeCall(cudaMalloc(&d_vertexPosTargetFloat3, sizeof(float3)*N));

            cutilSafeCall(cudaMalloc(&d_robustWeights, sizeof(float)*N));
			cutilSafeCall(cudaMalloc(&d_vertexPosFloat3, sizeof(float3)*N));
			cutilSafeCall(cudaMalloc(&d_vertexPosFloat3Urshape, sizeof(float3)*N));
			cutilSafeCall(cudaMalloc(&d_anglesFloat3, sizeof(float3)*N));
			cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
			cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
			cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));
            m_targetAccelerationStructure = generateAccelerationStructure(m_target);
			resetGPUMemory();   

            m_rnd       = std::mt19937(230948);
            m_uniform   = std::uniform_real_distribution<>(0,1);
		
            m_optWarpingSolver = new TerraWarpingSolver(N, 2 * E, d_neighbourIdx, d_neighbourOffset, "MeshDeformationAD.t", "gaussNewtonGPU");
		} 

        void setConstraints(float threshold = std::numeric_limits<float>::infinity(), float spuriousProbability = 0.1f)
		{
            
			unsigned int N = (unsigned int)m_result.n_vertices();
            std::uniform_int_distribution<> indexDistribution(0, N - 1);
			std::vector<float3> h_vertexPosTargetFloat3(N);
            float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            for (unsigned int i = 0; i < N; i++) {
                auto currentPt = m_result.point(VertexHandle(i));
                uint indexOfNearest = m_targetAccelerationStructure->nearest(currentPt.data());
                const Vec3f target = m_target.point(VertexHandle(indexOfNearest));

                bool spurious = m_uniform(m_rnd) < spuriousProbability;
                if (spurious) {
                    int spuriousIndex = indexDistribution(m_rnd);
                    const Vec3f spuriousTarget = m_target.point(VertexHandle(spuriousIndex));
                    h_vertexPosTargetFloat3[i] = make_float3(spuriousTarget[0], spuriousTarget[1], spuriousTarget[2]);
                } else {
                    if ((target - currentPt).length() < threshold) {
                        h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
                    }
                    else {
                        h_vertexPosTargetFloat3[i] = invalidPt;
                    }
                }
			}

			cutilSafeCall(cudaMemcpy(d_vertexPosTargetFloat3, h_vertexPosTargetFloat3.data(), sizeof(float3)*N, cudaMemcpyHostToDevice));
		}

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();
			unsigned int E = (unsigned int)m_initial.n_edges();

			float3* h_vertexPosFloat3 = new float3[N];
			int*	h_numNeighbours   = new int[N];
			int*	h_neighbourIdx	  = new int[2*E];
			int*	h_neighbourOffset = new int[N+1];
            float*  h_robustWeights = new float[N];

			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_initial.point(VertexHandle(i));
				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
                h_robustWeights[i] = 0.5f;
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
			setConstraints();


			// Angles
			float3* h_angles = new float3[N];
			for (unsigned int i = 0; i < N; i++)
			{
				h_angles[i] = make_float3(0.0f, 0.0f, 0.0f);
			}
			cutilSafeCall(cudaMemcpy(d_anglesFloat3, h_angles, sizeof(float3)*N, cudaMemcpyHostToDevice));
			delete [] h_angles;
			
			cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_vertexPosFloat3Urshape, h_vertexPosFloat3, sizeof(float3)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, sizeof(int)*N, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, sizeof(int)* 2 * E, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, sizeof(int)*(N + 1), cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy(d_robustWeights, h_robustWeights, sizeof(float)*N, cudaMemcpyHostToDevice));

            delete [] h_robustWeights;
			delete [] h_vertexPosFloat3;
			delete [] h_numNeighbours;
			delete [] h_neighbourIdx;
			delete [] h_neighbourOffset;
		}

		~ImageWarping()
		{
			cutilSafeCall(cudaFree(d_anglesFloat3));

            cutilSafeCall(cudaFree(d_robustWeights));

			cutilSafeCall(cudaFree(d_vertexPosTargetFloat3));
			cutilSafeCall(cudaFree(d_vertexPosFloat3));
			cutilSafeCall(cudaFree(d_vertexPosFloat3Urshape));
			cutilSafeCall(cudaFree(d_numNeighbours));
			cutilSafeCall(cudaFree(d_neighbourIdx));
			cutilSafeCall(cudaFree(d_neighbourOffset));

			SAFE_DELETE(m_optWarpingSolver);
		}

		SimpleMesh* solve()
		{
			
			//float weightFit = 6.0f;
            //float weightReg = 0.5f;
			float weightFit = 3.0f;
			float weightReg = 80.0f; //0.000001f;
		
			//unsigned int numIter = 10;
			//unsigned int nonLinearIter = 20;
			//unsigned int linearIter = 50;

			unsigned int numIter = 6;
			unsigned int nonLinearIter = 4;
            unsigned int linearIter = 1000;			

			m_result = m_initial;
			resetGPUMemory();
			for (unsigned int i = 0; i < numIter; i++)
			{
				std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
				setConstraints();

                m_optWarpingSolver->solveGN(d_vertexPosFloat3, d_anglesFloat3, d_robustWeights, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, nonLinearIter, linearIter, weightFit, weightReg);
                                #if EARLY_OUT
				break;
				#endif

			}
			copyResultToCPUFromFloat3();		
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

        std::unique_ptr<ml::NearestNeighborSearchFLANN<float>> generateAccelerationStructure(const SimpleMesh& mesh) {
            auto nnData = std::make_unique<ml::NearestNeighborSearchFLANN<float>>(50, 1);
            unsigned int N = (unsigned int)mesh.n_vertices();

            std::vector<const float*> flannPoints(N);
            for (unsigned int i = 0; i < N; i++)
            {   
                flannPoints[i] = mesh.point(VertexHandle(i)).data();
            }
            nnData->init(flannPoints, 3, 1);
            return nnData;
        }


        std::unique_ptr<ml::NearestNeighborSearchFLANN<float>> m_targetAccelerationStructure;

        std::mt19937 m_rnd;
        std::uniform_real_distribution<> m_uniform;

		SimpleMesh m_result;
		SimpleMesh m_initial;
        SimpleMesh m_target;
	
		float3* d_anglesFloat3;
		float3*	d_vertexPosTargetFloat3;
		float3*	d_vertexPosFloat3;
		float3*	d_vertexPosFloat3Urshape;
        float*  d_robustWeights;
		int*	d_numNeighbours;
		int*	d_neighbourIdx;
		int* 	d_neighbourOffset;

		TerraWarpingSolver* m_optWarpingSolver;
};

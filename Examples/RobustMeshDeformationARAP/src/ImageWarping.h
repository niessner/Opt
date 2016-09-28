#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"
#include "CudaArray.h"

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

            d_vertexPosTargetFloat3.alloc(N);
            d_robustWeights.alloc(N);
            d_vertexPosFloat3.alloc(N);
            d_vertexPosFloat3Urshape.alloc(N);
            d_anglesFloat3.alloc(N);
            d_numNeighbours.alloc(N);
            d_neighbourIdx.alloc(2*E);
            d_neighbourOffset.alloc(N + 1);
            m_targetAccelerationStructure = generateAccelerationStructure(m_target);
			resetGPUMemory();   

            m_rnd       = std::mt19937(230948);

            float spuriousProbability = 0.05f;

            std::uniform_int_distribution<> sourceDistribution(0, N - 1);
            std::uniform_int_distribution<> targetDistribution(0, targetMesh->n_vertices() - 1);
            int spuriousCount = int(N*spuriousProbability);
            for (int i = 0; i < spuriousCount; ++i) {
                m_spuriousIndexPairs.push_back(make_int2(sourceDistribution(m_rnd), targetDistribution(m_rnd)));
            }

            m_optWarpingSolver = new TerraWarpingSolver(N, 2 * E, d_neighbourIdx.data(), d_neighbourOffset.data(), "MeshDeformationAD.t", "gaussNewtonGPU");
		} 


        void setConstraints(float threshold = std::numeric_limits<float>::infinity())
		{
            
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
            float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            for (unsigned int i = 0; i < N; i++) {
                auto currentPt = m_result.point(VertexHandle(i));
                uint indexOfNearest = m_targetAccelerationStructure->nearest(currentPt.data());
                const Vec3f target = m_target.point(VertexHandle(indexOfNearest));

                if ((target - currentPt).length() < threshold) {
                    h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
                }
                else {
                    h_vertexPosTargetFloat3[i] = invalidPt;
                }
			}

            for (int2 iPair : m_spuriousIndexPairs) {
                const Vec3f spuriousTarget = m_target.point(VertexHandle(iPair.y));
                h_vertexPosTargetFloat3[iPair.x] = make_float3(spuriousTarget[0], spuriousTarget[1], spuriousTarget[2]);
            }
            d_vertexPosTargetFloat3.update(h_vertexPosTargetFloat3.data(), N);
		}

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();
			unsigned int E = (unsigned int)m_initial.n_edges();

            std::vector<float3> h_vertexPosFloat3(N);
            std::vector<int>	h_numNeighbours(N);
			std::vector<int>	h_neighbourIdx(2*E);
			std::vector<int>	h_neighbourOffset(N+1);
            std::vector<float>  h_robustWeights(N);

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
			std::vector<float3> h_angles(N);
			for (unsigned int i = 0; i < N; i++)
			{
				h_angles[i] = make_float3(0.0f, 0.0f, 0.0f);
			}
            d_anglesFloat3.update(h_angles.data(), N);
            d_vertexPosFloat3.update(h_vertexPosFloat3.data(), N);
            d_vertexPosFloat3Urshape.update(h_vertexPosFloat3.data(), N);
            d_numNeighbours.update(h_numNeighbours.data(), N);
            d_neighbourIdx.update(h_neighbourIdx.data(), 2 * E);
            d_neighbourOffset.update(h_neighbourOffset.data(), N + 1);
            d_robustWeights.update(h_robustWeights.data(), N);
		}

		~ImageWarping()
		{
			SAFE_DELETE(m_optWarpingSolver);
		}

		SimpleMesh* solve()
		{
			
			//float weightFit = 6.0f;
            //float weightReg = 0.5f;
			float weightFit = 10.0f;
			float weightReg = 160.0f; //0.000001f;
		
			//unsigned int numIter = 10;
			//unsigned int nonLinearIter = 20;
			//unsigned int linearIter = 50;

			unsigned int numIter = 20;
			unsigned int nonLinearIter = 4;
            unsigned int linearIter = 4000;			

			m_result = m_initial;
			resetGPUMemory();
			for (unsigned int i = 0; i < numIter; i++)
			{
				std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
                m_timer.start();
				setConstraints();
                m_timer.stop();
                double setConstraintsTime = m_timer.getElapsedTime();
                m_optWarpingSolver->solveGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), nonLinearIter, linearIter, weightFit, weightReg);
                // TODO: faster method to set constraints
                m_timer.start();
                copyResultToCPUFromFloat3();
                m_timer.stop();
                double copyTime = m_timer.getElapsedTime();
                std::cout << "-- Set Constraints: " << setConstraintsTime << "s -- Copy to CPU: " << copyTime << "s " << std::endl;

			}	
			return &m_result;
		}

		void copyResultToCPUFromFloat3()
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
            std::vector<float3> h_vertexPosFloat3(N);
			cutilSafeCall(cudaMemcpy(h_vertexPosFloat3.data(), d_vertexPosFloat3.data(), sizeof(float3)*N, cudaMemcpyDeviceToHost));

			for (unsigned int i = 0; i < N; i++)
			{
				m_result.set_point(VertexHandle(i), Vec3f(h_vertexPosFloat3[i].x, h_vertexPosFloat3[i].y, h_vertexPosFloat3[i].z));
			}
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

        ml::Timer m_timer;
        std::unique_ptr<ml::NearestNeighborSearchFLANN<float>> m_targetAccelerationStructure;

        std::mt19937 m_rnd;
        std::vector<int2> m_spuriousIndexPairs;

		SimpleMesh m_result;
		SimpleMesh m_initial;
        SimpleMesh m_target;
	
		CudaArray<float3> d_anglesFloat3;
		CudaArray<float3>	d_vertexPosTargetFloat3;
		CudaArray<float3>	d_vertexPosFloat3;
		CudaArray<float3>	d_vertexPosFloat3Urshape;
        CudaArray<float>  d_robustWeights;
		CudaArray<int>	d_numNeighbours;
		CudaArray<int>	d_neighbourIdx;
		CudaArray<int> 	d_neighbourOffset;

		TerraWarpingSolver* m_optWarpingSolver;
};

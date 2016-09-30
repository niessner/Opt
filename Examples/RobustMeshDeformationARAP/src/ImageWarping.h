#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"
#include "CudaArray.h"

static bool operator==(const float3& v0, const float3& v1) {
    return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool operator!=(const float3& v0, const float3& v1) {
    return !(v0 == v1);
}

class ImageWarping
{
	public:
        ImageWarping(const SimpleMesh* sourceMesh, const std::vector<SimpleMesh*>& targetMeshes)
		{
            m_result = *sourceMesh;
			m_initial = m_result;

            for (SimpleMesh* mesh : targetMeshes) {
                m_targets.push_back(*mesh);
            }

            unsigned int N = (unsigned int)sourceMesh->n_vertices();
            unsigned int E = (unsigned int)sourceMesh->n_edges();
            
            double sumEdgeLength = 0.0f;
            for (auto edgeHandle : m_initial.edges()) {
                auto edge = m_initial.edge(edgeHandle);
                sumEdgeLength += m_initial.calc_edge_length(edgeHandle);
            }
            m_averageEdgeLength = sumEdgeLength / E;
            std::cout << "Average Edge Length: " << m_averageEdgeLength << std::endl;

            d_vertexPosTargetFloat3.alloc(N);
            d_vertexNormalTargetFloat3.alloc(N);
            d_robustWeights.alloc(N);
            d_vertexPosFloat3.alloc(N);
            d_vertexPosFloat3Urshape.alloc(N);
            d_anglesFloat3.alloc(N);
            d_numNeighbours.alloc(N);
            d_neighbourIdx.alloc(2*E);
            d_neighbourOffset.alloc(N + 1);
            
			resetGPUMemory();   

            m_rnd       = std::mt19937(230948);

            float spuriousProbability = 0.0f;

            std::uniform_int_distribution<> sourceDistribution(0, N - 1);
            std::uniform_int_distribution<> targetDistribution(0, m_targets[0].n_vertices() - 1);
            int spuriousCount = int(N*spuriousProbability);
            for (int i = 0; i < spuriousCount; ++i) {
                m_spuriousIndexPairs.push_back(make_int2(sourceDistribution(m_rnd), targetDistribution(m_rnd)));
            }

            m_optWarpingSolver = new TerraWarpingSolver(N, 2 * E, d_neighbourIdx.data(), d_neighbourOffset.data(), "MeshDeformationAD.t", "gaussNewtonGPU");
		} 




        int setConstraints(int targetIndex, float positionThreshold = std::numeric_limits<float>::infinity(), float cosNormalThreshold = 0.7f)
		{
            
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
            std::vector<float3> h_vertexNormalTargetFloat3(N);

            uint N_target = m_targets[targetIndex].n_vertices();
            std::vector<float> smallestDist(N_target);

            for (int i = 0; i < N_target; ++i) {
                smallestDist[i] = std::numeric_limits<float>::infinity();
            }

            if (!(m_targets[targetIndex].has_face_normals() && m_targets[targetIndex].has_vertex_normals())) { 
                m_targets[targetIndex].request_face_normals();
                m_targets[targetIndex].request_vertex_normals();
                // let the mesh update the normals
                m_targets[targetIndex].update_normals();
            }

            if (!(m_result.has_face_normals() && m_result.has_vertex_normals())) {
                m_result.request_face_normals();
                m_result.request_vertex_normals();
            }
            m_result.update_normals();

            uint thrownOutCorrespondenceCount = 0;
            float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            for (unsigned int i = 0; i < N; i++) {
                auto currentPt = m_result.point(VertexHandle(i));
                uint indexOfNearest = m_targetAccelerationStructure->nearest(currentPt.data());
                const Vec3f target = m_targets[targetIndex].point(VertexHandle(indexOfNearest));
                auto tNormal = m_targets[targetIndex].normal(VertexHandle(indexOfNearest));
                auto targetNormal = make_float3(tNormal[0], tNormal[1], tNormal[2]);
                auto sNormal = m_result.normal(VertexHandle(i));
                auto sourceNormal = make_float3(sNormal[0], sNormal[1], sNormal[2]);
                float dist = (target - currentPt).length();
                if (dist < positionThreshold && dist < smallestDist[indexOfNearest] &&
                    dot(targetNormal, sourceNormal) > cosNormalThreshold) {
                    smallestDist[indexOfNearest] = dist; // Comment out to allow multiple points to target the same mesh
                    h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
                } else {
                    ++thrownOutCorrespondenceCount;
                    h_vertexPosTargetFloat3[i] = invalidPt;
                }
                
                h_vertexNormalTargetFloat3[i] = targetNormal;
			}

            for (int2 iPair : m_spuriousIndexPairs) {
                const Vec3f spuriousTarget = m_targets[targetIndex].point(VertexHandle(iPair.y));
                h_vertexPosTargetFloat3[iPair.x] = make_float3(spuriousTarget[0], spuriousTarget[1], spuriousTarget[2]);
                auto normal = m_targets[targetIndex].normal(VertexHandle(iPair.y));
                h_vertexNormalTargetFloat3[iPair.x] = make_float3(normal[0], normal[1], normal[2]);
            }
            d_vertexPosTargetFloat3.update(h_vertexPosTargetFloat3.data(), N);
            d_vertexNormalTargetFloat3.update(h_vertexNormalTargetFloat3.data(), N);

            std::vector<float>  h_robustWeights(N);
            cutilSafeCall(cudaMemcpy(h_robustWeights.data(), d_robustWeights.data(), sizeof(float)*N, cudaMemcpyDeviceToHost));

            int constraintsUpdated = 0;
            for (int i = 0; i < N; ++i) {
                if (m_previousConstraints[i] != h_vertexPosTargetFloat3[i]) {
                    m_previousConstraints[i] = h_vertexPosTargetFloat3[i];
                    ++constraintsUpdated;
                    {
                        auto currentPt = m_result.point(VertexHandle(i));
                        auto v = h_vertexPosTargetFloat3[i];
                        const Vec3f target = Vec3f(v.x, v.y, v.z);
                        float dist = (target - currentPt).length();
                        float weight = (positionThreshold - dist) / positionThreshold;
                        h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.5f);
                    }
                }
            }
            d_robustWeights.update(h_robustWeights.data(), N);

            std::cout << "*******Thrown out correspondence count: " << thrownOutCorrespondenceCount << std::endl;

            return constraintsUpdated;
		}

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();
			unsigned int E = (unsigned int)m_initial.n_edges();

            std::vector<float3> h_vertexPosFloat3(N);
            std::vector<int>	h_numNeighbours(N);
			std::vector<int>	h_neighbourIdx(2*E);
			std::vector<int>	h_neighbourOffset(N+1);

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

			uint numIter = 40;
			uint nonLinearIter = 16;
            uint linearIter = 1000;			
            unsigned int N = (unsigned int)m_initial.n_vertices();

			m_result = m_initial;
			resetGPUMemory();
            for (uint targetIndex = 0; targetIndex < m_targets.size(); ++targetIndex) {
                m_timer.start();
                m_targetAccelerationStructure = generateAccelerationStructure(m_targets[targetIndex]);
                m_timer.stop();
                m_previousConstraints.resize(N);
                for (int i = 0; i < N; ++i) {
                    m_previousConstraints[i] = make_float3(0, 0, -90901283092183);
                }
                std::cout << "---- Acceleration Structure Build: " << m_timer.getElapsedTime() << "s" << std::endl;
                for (uint i = 0; i < numIter; i++)
                {
                    std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;

                    m_timer.start();
                    int newConstraintCount = setConstraints(targetIndex, m_averageEdgeLength*3.0f);
                    m_timer.stop();
                    double setConstraintsTime = m_timer.getElapsedTime();
                    
                    std::cout << " -------- New constraints: " << newConstraintCount << std::endl;
                    if (newConstraintCount < 1) {
                        std::cout << "VERY LITTLE CHANGE, EARLY OUT" << std::endl;
                        break;
                    }
                    m_optWarpingSolver->solveGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), d_vertexNormalTargetFloat3.data(), nonLinearIter, linearIter, weightFit, weightReg);
                    // TODO: faster method to set constraints

                    m_timer.start();
                    copyResultToCPUFromFloat3();
                    m_timer.stop();
                    double copyTime = m_timer.getElapsedTime();
                    std::cout << "-- Set Constraints: " << setConstraintsTime << "s -- Copy to CPU: " << copyTime << "s " << std::endl;

                }
                { // Save intermediate mesh
                    char buff[100];
                    sprintf(buff, "out_%04d.ply", targetIndex);
                    int failure = OpenMesh::IO::write_mesh(m_result, buff);
                    assert(failure);
                }
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
        std::vector<SimpleMesh> m_targets;
        std::vector<float3> m_previousConstraints;
	
        double m_averageEdgeLength;

		CudaArray<float3> d_anglesFloat3;
		CudaArray<float3>	d_vertexPosTargetFloat3;
        CudaArray<float3>	d_vertexNormalTargetFloat3;
		CudaArray<float3>	d_vertexPosFloat3;
		CudaArray<float3>	d_vertexPosFloat3Urshape;
        CudaArray<float>  d_robustWeights;
		CudaArray<int>	d_numNeighbours;
		CudaArray<int>	d_neighbourIdx;
		CudaArray<int> 	d_neighbourOffset;

		TerraWarpingSolver* m_optWarpingSolver;
};

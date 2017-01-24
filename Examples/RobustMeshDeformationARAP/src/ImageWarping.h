#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/SolverIteration.h"

#include "TerraWarpingSolver.h"
#include "OpenMesh.h"
#include "CudaArray.h"

// From the future (C++14)
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


static bool operator==(const float3& v0, const float3& v1) {
    return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool operator!=(const float3& v0, const float3& v1) {
    return !(v0 == v1);
}
#define MAX_K 20


class ImageWarping
{

	public:
        ImageWarping(const SimpleMesh* sourceMesh, const std::vector<SimpleMesh*>& targetMeshes, const std::vector<int4>& sourceTetIndices, int startIndex)
		{
            m_result = *sourceMesh;
			m_initial = m_result;
            m_sourceTetIndices = sourceTetIndices;
            m_startIndex = startIndex;

            for (SimpleMesh* mesh : targetMeshes) {
                m_targets.push_back(*mesh);
            }

            uint N = (uint)sourceMesh->n_vertices();
            uint E = (uint)sourceMesh->n_edges();
            
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
            
			resetGPUMemory();   

            m_rnd       = std::mt19937(230948);

            float spuriousProbability = 0.05f;

            std::uniform_int_distribution<> targetDistribution(0, m_targets[0].n_vertices() - 1);
            float noiseModifier = 30.0f;
            std::normal_distribution<> normalDistribution(0.0f, m_averageEdgeLength * noiseModifier);
            int spuriousCount = int(N*spuriousProbability);
            for (int i = 0; i < spuriousCount; ++i) {
                m_spuriousIndices.push_back(targetDistribution(m_rnd));
                m_noisyOffsets.push_back(make_float3(normalDistribution(m_rnd), normalDistribution(m_rnd), normalDistribution(m_rnd)));
            }


            if (!m_result.has_vertex_colors()) {
                m_result.request_vertex_colors();
            }
            for (unsigned int i = 0; i < N; i++)
            {
                uchar w = 255;
                m_result.set_color(VertexHandle(i), Vec3uc(w, w, w));
            }
            for (int index : m_spuriousIndices) {
                uchar w = 0;
                //m_result.set_color(VertexHandle(index), Vec3uc(w, w, w));
            }

            OpenMesh::IO::Options options = OpenMesh::IO::Options::VertexColor;
            int failure = OpenMesh::IO::write_mesh(m_result, "out_noisetemplate.ply", options);
            assert(failure);



            m_optWarpingSolver = new TerraWarpingSolver(N, d_neighbourIdx.size(), d_neighbourIdx.data(), d_neighbourOffset.data(), "MeshDeformationAD.t", "gaussNewtonGPU");
		} 


        inline vec3f convertDepthToRGB(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
            float depthZeroOne = (depth - depthMin) / (depthMax - depthMin);
            float x = 1.0f - depthZeroOne;
            if (x < 0.0f) x = 0.0f;
            if (x > 1.0f) x = 1.0f;
            return BaseImageHelper::convertHSVtoRGB(vec3f(240.0f*x, 1.0f, 0.5f));
        }

        void saveCurrentMesh(std::string filename) {
            { // Save intermediate mesh
                unsigned int N = (unsigned int)m_result.n_vertices();
                std::vector<float> h_vertexWeightFloat(N);
                cutilSafeCall(cudaMemcpy(h_vertexWeightFloat.data(), d_robustWeights.data(), sizeof(float)*N, cudaMemcpyDeviceToHost));
                for (unsigned int i = 0; i < N; i++)
                {
                    vec3f color = convertDepthToRGB(1.0f-clamp(h_vertexWeightFloat[i], 0.0f, 1.0f));

                    m_result.set_color(VertexHandle(i), Vec3uc(color.r * 255, color.g * 255, color.b * 255));

                    if (h_vertexWeightFloat[i] < 0.9f || h_vertexWeightFloat[i] > 1.0f) {
                        printf("Interesting robustWeight[%d]: %f\n", i, h_vertexWeightFloat[i]);
                    }
                }
                
                OpenMesh::IO::Options options = OpenMesh::IO::Options::VertexColor;
                int failure = OpenMesh::IO::write_mesh(m_result, filename, options);
                assert(failure);
            }
        }


        int setConstraints(int targetIndex, float positionThreshold = std::numeric_limits<float>::infinity(), float cosNormalThreshold = 0.7f)
		{
            
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
            std::vector<float3> h_vertexNormalTargetFloat3(N);

            uint N_target = m_targets[targetIndex].n_vertices();

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

//#pragma omp parallel for // TODO: check why it makes everything wrongs
            for (int i = 0; i < (int)N; i++) {
                std::vector<uint> neighbors(MAX_K);
                auto currentPt = m_result.point(VertexHandle(i));
                auto sNormal = m_result.normal(VertexHandle(i));
                auto sourceNormal = make_float3(sNormal[0], sNormal[1], sNormal[2]);
                m_targetAccelerationStructure->kNearest(currentPt.data(), MAX_K, 0.0f, neighbors);
                bool validTargetFound = false;
                for (uint indexOfNearest : neighbors) {
                    const Vec3f target = m_targets[targetIndex].point(VertexHandle(indexOfNearest));
                    auto tNormal = m_targets[targetIndex].normal(VertexHandle(indexOfNearest));
                    auto targetNormal = make_float3(tNormal[0], tNormal[1], tNormal[2]);
                    float dist = (target - currentPt).length();
                    if (dist > positionThreshold) {
                        break;
                    }
                    if (dot(targetNormal, sourceNormal) > cosNormalThreshold) {
                        h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
                        h_vertexNormalTargetFloat3[i] = targetNormal;
                        validTargetFound = true;
                        break;
                    }
                    
                }
                if (!validTargetFound) {
                    ++thrownOutCorrespondenceCount;
                    h_vertexPosTargetFloat3[i] = invalidPt;
                }
			}

            for (int i = 0; i < m_spuriousIndices.size(); ++i) {
                h_vertexPosTargetFloat3[m_spuriousIndices[i]] += m_noisyOffsets[i];
                /*float3 before = *((float3*)mesh.point(VertexHandle(i)).data());
                *((float3*)mesh.point(VertexHandle(i)).data()) += m_noisyOffsets[i];
                float3 after = *((float3*)mesh.point(VertexHandle(i)).data());*/
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
                        h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f);
                        h_robustWeights[i] = 1.0f;
                    }
                }
            }
            d_robustWeights.update(h_robustWeights.data(), N);

            std::cout << "*******Thrown out correspondence count: " << thrownOutCorrespondenceCount << std::endl;

            return constraintsUpdated;
		}

        void generateOptEdges(std::vector<int>& h_numNeighbours, std::vector<int>&	h_neighbourIdx, std::vector<int>& h_neighbourOffset) {
            uint N = (uint)m_initial.n_vertices();
            h_numNeighbours.resize(N);
            h_neighbourOffset.resize(N + 1);
            if (m_sourceTetIndices.size() == 0) {
                uint E = (uint)m_initial.n_edges();
                h_neighbourIdx.resize(2 * E);
                


                unsigned int count = 0;
                unsigned int offset = 0;
                h_neighbourOffset[0] = 0;
                for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
                {
                    VertexHandle c_vh(v_it.handle());
                    unsigned int valence = m_initial.valence(c_vh);
                    h_numNeighbours[count] = valence;
                    for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it; vv_it++)
                    {
                        VertexHandle v_vh(vv_it.handle());
                        h_neighbourIdx[offset] = v_vh.idx();
                        offset++;
                    }

                    h_neighbourOffset[count + 1] = offset;

                    count++;
                }
            } else {
                // Potentially very inefficient. I don't care right now
                std::vector<std::set<int>> neighbors(N);
                for (int4 tet : m_sourceTetIndices) {
                    int* t = (int*)&tet;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 1; j < 4; ++j) {
                            neighbors[t[i]].insert(t[(i + j) % 4]);
                        }
                    }
                }
                uint offset = 0;
                h_neighbourOffset[0] = 0;
                for (int i = 0; i < N; ++i) {
                    int valence = neighbors[i].size();
                    h_numNeighbours[i] = valence;
                    for (auto n : neighbors[i]) {
                        h_neighbourIdx.push_back(n);
                        ++offset;
                    }
                    h_neighbourOffset[i+1] = offset;

                }
            }
            printf("Total Edge count = %d\n", h_neighbourIdx.size());
        }

		void resetGPUMemory()
		{
            std::vector<int> h_numNeighbours, h_neighbourIdx, h_neighbourOffset;
            generateOptEdges(h_numNeighbours, h_neighbourIdx, h_neighbourOffset);
            uint N = (uint)m_initial.n_vertices();
            std::vector<float3> h_vertexPosFloat3(N);

			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_initial.point(VertexHandle(i));
				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
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
            d_neighbourIdx.update(h_neighbourIdx.data(), h_neighbourIdx.size());
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
			float weightRegMax = 64.0f; 

            float weightRegMin = 4.0f;
            float weightRegFactor = 0.95f;

            float costEqualityEpsilon = 0.00001f;
		
			//unsigned int numIter = 10;
			//unsigned int nonLinearIter = 20;
			//unsigned int linearIter = 50;

			m_params.numIter = 15;
			m_params.nonLinearIter = 8;
            m_params.linearIter = 250;			
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
                float weightReg = weightRegMax;
                for (uint i = 0; i < m_params.numIter; i++)
                {
                    std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
                    
                    m_timer.start();
                    int newConstraintCount = setConstraints(targetIndex, m_averageEdgeLength*5.0f);


                    m_timer.stop();
                    double setConstraintsTime = m_timer.getElapsedTime();
                    
                    std::cout << " -------- New constraints: " << newConstraintCount << std::endl;
                    if (newConstraintCount <= 5) {
                        std::cout << " -------- Few New Constraints" << std::endl;
                        if (weightReg != weightRegMin) {
                            std::cout << " -------- Skipping to min reg weight" << std::endl;
                            weightReg = weightRegMin;
                            m_optWarpingSolver->initGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), d_vertexNormalTargetFloat3.data(), m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg);
                            float currentCost = m_optWarpingSolver->currentCost();
                            for (int nlIter = 0; nlIter <= m_params.nonLinearIter; ++nlIter) {
                                m_optWarpingSolver->stepGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), d_vertexNormalTargetFloat3.data(), m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg);
                                float newCost = m_optWarpingSolver->currentCost();
                                if ((currentCost - newCost) < costEqualityEpsilon) {
                                    std::cout << " -------- NO SIGNIFICANT COST CHANGE, FINISHING OUTER-ITER EARLY" << std::endl;
                                    break;
                                }
                                else {
                                    currentCost = newCost;
                                }
                            }
                        }
                        break;
                    }
                    m_optWarpingSolver->initGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), d_vertexNormalTargetFloat3.data(), m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg);
                    float currentCost = m_optWarpingSolver->currentCost();
                    for (int nlIter = 0; nlIter <= m_params.nonLinearIter; ++nlIter) {
                        m_optWarpingSolver->stepGN(d_vertexPosFloat3.data(), d_anglesFloat3.data(), d_robustWeights.data(), d_vertexPosFloat3Urshape.data(), d_vertexPosTargetFloat3.data(), d_vertexNormalTargetFloat3.data(), m_params.nonLinearIter, m_params.linearIter, weightFit, weightReg);
                        float newCost = m_optWarpingSolver->currentCost();
                        if ((currentCost - newCost) < costEqualityEpsilon) {
                            std::cout << " -------- NO SIGNIFICANT COST CHANGE, FINISHING OUTER-ITER EARLY" << std::endl;
                            break;
                        } else {
                            currentCost = newCost;
                        }
                        weightReg = fmaxf(weightRegMin, weightReg*weightRegFactor);
                    }
                    m_timer.start();
                    copyResultToCPUFromFloat3();
                    m_timer.stop();
                    double copyTime = m_timer.getElapsedTime();
                    std::cout << "-- Set Constraints: " << setConstraintsTime << "s -- Copy to CPU: " << copyTime << "s " << std::endl;
                   /* if (targetIndex + m_startIndex + 1 == 13) {
                        char buff[100];
                        sprintf(buff, "out_%04d_%02d.ply", 13, i);
                        saveCurrentMesh(buff);
                    }*/
                }
                char buff[100];
                sprintf(buff, "out_%04d.ply", targetIndex + m_startIndex + 1);
                saveCurrentMesh(buff);
            }
            reportFinalCosts("Robust Mesh Deformation", m_params, m_optWarpingSolver->currentCost(), nan(nullptr), nan(nullptr));

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
            auto nnData = make_unique<ml::NearestNeighborSearchFLANN<float>>(max(50, 4 * MAX_K), 1);
            unsigned int N = (unsigned int)mesh.n_vertices();

            assert(m_spuriousIndices.size() == m_noisyOffsets.size());
            /*
            for (int i = 0; i < m_spuriousIndices.size(); ++i) {
                float3 before = *((float3*)mesh.point(VertexHandle(i)).data());
                *((float3*)mesh.point(VertexHandle(i)).data()) += m_noisyOffsets[i];
                float3 after = *((float3*)mesh.point(VertexHandle(i)).data());
            }*/

            std::vector<const float*> flannPoints(N);
            for (unsigned int i = 0; i < N; i++)
            {   
                flannPoints[i] = mesh.point(VertexHandle(i)).data();
            }
            nnData->init(flannPoints, 3, MAX_K);
            return nnData;
        }


        ml::Timer m_timer;
        std::unique_ptr<ml::NearestNeighborSearchFLANN<float>> m_targetAccelerationStructure;

        int m_startIndex;

        std::mt19937 m_rnd;
        std::vector<int> m_spuriousIndices;;
        std::vector<float3> m_noisyOffsets;

		SimpleMesh m_result;
		SimpleMesh m_initial;
        std::vector<SimpleMesh> m_targets;
        std::vector<float3> m_previousConstraints;
        std::vector<int4> m_sourceTetIndices;

        double m_averageEdgeLength;

		CudaArray<float3> d_anglesFloat3;
		CudaArray<float3> d_vertexPosTargetFloat3;
        CudaArray<float3> d_vertexNormalTargetFloat3;
		CudaArray<float3> d_vertexPosFloat3;
		CudaArray<float3> d_vertexPosFloat3Urshape;
        CudaArray<float>  d_robustWeights;
		CudaArray<int>	d_numNeighbours;
		CudaArray<int>	d_neighbourIdx;
		CudaArray<int> 	d_neighbourOffset;

        CombinedSolverParameters m_params;

		TerraWarpingSolver* m_optWarpingSolver;
};

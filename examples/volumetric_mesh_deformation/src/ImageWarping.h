#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAWarpingSolver.h"
#include "CERESWarpingSolver.h"
#include "OpenMesh.h"

#include "../../shared/OptSolver.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/SolverIteration.h"

/*
class CombinedSolverBase {
public:
    virtual void preSolve
protected:
    NamedParameters m_solverParams;
    NamedParameters m_problemParams;
};*/

class ImageWarping
{
	public:
		ImageWarping(const SimpleMesh* mesh)
		{
			m_result = *mesh;
			m_initial = m_result;

			m_dims = make_int3(10, 40, 10);
			m_nNodes = (m_dims.x + 1)*(m_dims.y + 1)*(m_dims.z + 1);
			
			unsigned int N = (unsigned int)mesh->n_vertices();
		
			cutilSafeCall(cudaMalloc(&d_gridPosTargetFloat3, sizeof(float3)*m_nNodes));
			cutilSafeCall(cudaMalloc(&d_gridPosFloat3, sizeof(float3)*m_nNodes));
			cutilSafeCall(cudaMalloc(&d_gridPosFloat3Urshape, sizeof(float3)*m_nNodes));
			cutilSafeCall(cudaMalloc(&d_gridAnglesFloat3, sizeof(float3)*m_nNodes));

			m_vertexToVoxels = new int3[N];
			m_relativeCoords = new float3[N];

			resetGPUMemory();
            std::vector<unsigned int> dims = { (uint)m_dims.x + 1, (uint)m_dims.y + 1, (uint)m_dims.z + 1 };
			m_warpingSolver         = std::make_unique<CUDAWarpingSolver>(dims);
            m_warpingSolverCeres    = std::make_unique<CeresSolver>(dims);
            m_warpingSolverOpt      = std::make_unique<OptSolver>(dims, "volumetric_mesh_deformation.t", "gaussNewtonGPU");
            m_warpingSolverOptLM    = std::make_unique<OptSolver>(dims, "volumetric_mesh_deformation.t", "LMGPU");
		}

		void setConstraints(float alpha)
		{
			float3* h_gridPosTargetFloat3 = new float3[m_nNodes];
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						int index = getIndex1D(make_int3(i, j, k));
						vec3f delta(m_delta.x, m_delta.y, m_delta.z);
						mat3f fac = mat3f::diag((float)i, (float)j, (float)k);
						vec3f min(m_min.x, m_min.y, m_min.z);
						vec3f v = min + fac*delta;

						if (j == 0) { h_gridPosTargetFloat3[index] = make_float3(v.x, v.y, v.z); }
						else if (j == m_dims.y)	{
							mat3f f = mat3f::diag(m_dims.x / 2.0f, (float)m_dims.y, m_dims.z / 2.0f);
							vec3f mid = vec3f(m_min.x, m_min.y, m_min.z) + f*delta; mat3f R = ml::mat3f::rotationZ(-90.0f);
							v = R*(v - mid) + mid + vec3f(2.5f, -2.5f, 0.0f);
							h_gridPosTargetFloat3[index] = make_float3(v.x, v.y, v.z);
						}
						else { h_gridPosTargetFloat3[index] = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()); }
					}
				}
			}

			cutilSafeCall(cudaMemcpy(d_gridPosTargetFloat3, h_gridPosTargetFloat3, sizeof(float3)*m_nNodes, cudaMemcpyHostToDevice));
			delete [] h_gridPosTargetFloat3;
		}

		void computeBoundingBox()
		{
			m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
				SimpleMesh::Point p = m_initial.point(VertexHandle(v_it.handle()));
				m_min.x = fmin(m_min.x, p[0]); m_min.y = fmin(m_min.y, p[1]); m_min.z = fmin(m_min.z, p[2]);
				m_max.x = fmax(m_max.x, p[0]); m_max.y = fmax(m_max.y, p[1]); m_max.z = fmax(m_max.z, p[2]);
			}
		}

		void GraphAddSphere(SimpleMesh& out, SimpleMesh::Point offset, float scale, SimpleMesh meshSphere, bool constrained)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshSphere.n_vertices(); i++)
			{
				SimpleMesh::Point p = meshSphere.point(VertexHandle(i))*scale + offset;
				VertexHandle vh = out.add_vertex(p);
				out.set_color(vh, SimpleMesh::Color(200, 0, 0));
			}

			for (unsigned int k = 0; k < meshSphere.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshSphere.fv_begin(FaceHandle(k)); v_it != meshSphere.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void GraphAddCone(SimpleMesh& out, SimpleMesh::Point offset, float scale, float3 direction, SimpleMesh meshCone)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshCone.n_vertices(); i++)
			{
				SimpleMesh::Point pO = meshCone.point(VertexHandle(i));

				vec3f o(0.0f, 0.5f, 0.0f);
				mat3f s = ml::mat3f::diag(scale, length(direction), scale);
				vec3f p(pO[0], pO[1], pO[2]);

				p = s*(p + o);

				vec3f f(direction.x, direction.y, direction.z); f = ml::vec3f::normalize(f);
				vec3f up(0.0f, 1.0f, 0.0f);
				vec3f axis  = ml::vec3f::cross(up, f);
				float angle = acos(ml::vec3f::dot(up, f))*180.0f/(float)PI;
				mat3f R = ml::mat3f::rotation(axis, angle); if (axis.length() < 0.00001f) R = ml::mat3f::identity();
				
				p = R*p;

				VertexHandle vh = out.add_vertex(SimpleMesh::Point(p.x, p.y, p.z) + offset);
				out.set_color(vh, SimpleMesh::Color(70, 200, 70));
			}

			for (unsigned int k = 0; k < meshCone.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshCone.fv_begin(FaceHandle(k)); v_it != meshCone.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void saveGraph(const std::string& filename, float3* data, unsigned int N, float scale, SimpleMesh meshSphere, SimpleMesh meshCone)
		{
			SimpleMesh out;

			float3* h_gridPosTarget = new float3[m_nNodes];
			cutilSafeCall(cudaMemcpy(h_gridPosTarget, d_gridPosTargetFloat3, sizeof(float3)*m_nNodes, cudaMemcpyDeviceToHost));
			for (unsigned int i = 0; i < N; i++)
			{
				if (h_gridPosTarget[i].x != -std::numeric_limits<float>::infinity())
				{
					GraphAddSphere(out, SimpleMesh::Point(data[i].x, data[i].y, data[i].z), scale, meshSphere, h_gridPosTarget[i].x != -std::numeric_limits<float>::infinity());
				}
			}
			delete[] h_gridPosTarget;

			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						float3 pos0 = data[getIndex1D(make_int3(i, j, k))];
						
						if (i + 1 <= m_dims.x) { float3 dir1 = data[getIndex1D(make_int3(i + 1, j, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir1, meshCone); }
						if (j + 1 <= m_dims.y) { float3 dir2 = data[getIndex1D(make_int3(i, j + 1, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir2, meshCone); }
						if (k + 1 <= m_dims.z) { float3 dir3 = data[getIndex1D(make_int3(i, j, k + 1))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir3, meshCone); }
					}
				}
			}

			OpenMesh::IO::write_mesh(out, filename, IO::Options::VertexColor);
		}

		void initializeWarpGrid()
		{
			float3* h_gridVertexPosFloat3 = new float3[m_nNodes];
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						float3 fac = make_float3((float)i, (float)j, (float)k);
						float3 v = m_min + fac*m_delta;
						h_gridVertexPosFloat3[getIndex1D(make_int3(i, j, k))] = v;
					}
				}
			}

			cutilSafeCall(cudaMemcpy(d_gridPosFloat3,		 h_gridVertexPosFloat3, sizeof(float3)*m_nNodes, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(d_gridPosFloat3Urshape, h_gridVertexPosFloat3, sizeof(float3)*m_nNodes, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemset(d_gridAnglesFloat3,	 0, sizeof(float3)*m_nNodes));

			delete [] h_gridVertexPosFloat3;
		}

		void resetGPUMemory()
		{
			computeBoundingBox();

			float EPS = 0.000001f;
			m_min -= make_float3(EPS, EPS, EPS);
			m_max += make_float3(EPS, EPS, EPS);
			m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);

			initializeWarpGrid();

			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
			    VertexHandle c_vh(v_it.handle());
				SimpleMesh::Point p = m_initial.point(c_vh);
				float3 pp = make_float3(p[0], p[1], p[2]);

				pp = (pp - m_min);
				pp.x /= m_delta.x;
				pp.y /= m_delta.y;
				pp.z /= m_delta.z;

				int3 pInt = make_int3((int)pp.x, (int)pp.y, (int)pp.z);
				m_vertexToVoxels[c_vh.idx()] = pInt;
				m_relativeCoords[c_vh.idx()] = pp - make_float3((float)pInt.x, (float)pInt.y, (float)pInt.z);
			}

			// Constraints
			setConstraints(1.0f);
		}

		~ImageWarping()
		{

			cutilSafeCall(cudaFree(d_gridPosTargetFloat3));
			cutilSafeCall(cudaFree(d_gridPosFloat3));
			cutilSafeCall(cudaFree(d_gridPosFloat3Urshape));
			cutilSafeCall(cudaFree(d_gridAnglesFloat3));

			delete [] m_vertexToVoxels;
			delete [] m_relativeCoords;

		}

		SimpleMesh* solve()
		{
			float weightFit = 1.0f;
			float weightReg = 0.05f;

            m_params.useCUDA = true;
            m_params.useOptLM = true;
            m_params.useCeres = true;

			m_params.nonLinearIter = 20;
            m_params.linearIter = 60;

            float weightFitSqrt = sqrtf(weightFit);
            float weightRegSqrt = sqrtf(weightReg);

            NamedParameters probParams;
            probParams.set("Offset", d_gridPosFloat3);
            probParams.set("Angle", d_gridAnglesFloat3);
            probParams.set("UrShape", d_gridPosFloat3Urshape);
            probParams.set("Constraints", d_gridPosTargetFloat3);
            probParams.set("w_fitSqrt", &weightFitSqrt);
            probParams.set("w_regSqrt", &weightRegSqrt);

            NamedParameters solverParams;
            solverParams.set("nonLinearIterations", &m_params.nonLinearIter);
            solverParams.set("linearIterations", &m_params.linearIter);

			if (m_params.useCUDA)
			{
				m_result = m_initial;
				resetGPUMemory();
				std::cout << "//////////// (CUDA) ///////////////" << std::endl;
                m_warpingSolver->solve(solverParams, probParams, true, m_optIters);

				copyResultToCPUFromFloat3();
			}
			if (m_params.useOpt)
			{
				m_result = m_initial;
				resetGPUMemory();
				std::cout << "//////////// (OPT GN) ///////////////" << std::endl;
                m_warpingSolverOpt->solve(solverParams, probParams, true, m_optIters);
				copyResultToCPUFromFloat3();
			}

            if (m_params.useOptLM)
            {
                m_result = m_initial;
                resetGPUMemory();
                std::cout << "//////////// (OPT LM) ///////////////" << std::endl;
                m_warpingSolverOptLM->solve(solverParams, probParams, true, m_optLMIters);
                copyResultToCPUFromFloat3();
            }

			if (m_params.useCeres)
			{
				m_result = m_initial;
				resetGPUMemory();
				std::cout << "//////////// (CERES) ///////////////" << std::endl;
                m_warpingSolverCeres->solve(solverParams, probParams, true, m_ceresIters);

				copyResultToCPUFromFloat3();
			}

            saveSolverResults("results/", OPT_DOUBLE_PRECISION ? "_double" : "_float", m_ceresIters, m_optIters, m_optLMIters);

            reportFinalCosts("Mesh Deformation LARAP", m_params, m_warpingSolverOpt->finalCost(), m_warpingSolverOptLM->finalCost(), m_warpingSolverCeres->finalCost());

						
			return &m_result;
		}

		int getIndex1D(int3 idx)
        {
			return idx.x*((m_dims.y + 1)*(m_dims.z + 1)) + idx.y*(m_dims.z + 1) + idx.z;
		}

		void saveGraphResults()
		{
			SimpleMesh meshSphere;
			if (!OpenMesh::IO::read_mesh(meshSphere, "meshes/sphere.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			SimpleMesh meshCone;
			if (!OpenMesh::IO::read_mesh(meshCone, "meshes/cone.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			float3* h_gridPosUrshapeFloat3 = new float3[m_nNodes];
			cutilSafeCall(cudaMemcpy(h_gridPosUrshapeFloat3, d_gridPosFloat3Urshape, sizeof(float3)*m_nNodes, cudaMemcpyDeviceToHost));
			saveGraph("grid.ply", h_gridPosUrshapeFloat3, m_nNodes, 0.05f, meshSphere, meshCone);
			delete[] h_gridPosUrshapeFloat3;

			float3* h_gridPosFloat3 = new float3[m_nNodes];
			cutilSafeCall(cudaMemcpy(h_gridPosFloat3, d_gridPosFloat3, sizeof(float3)*m_nNodes, cudaMemcpyDeviceToHost));
			saveGraph("gridOut.ply", h_gridPosFloat3, m_nNodes, 0.05f, meshSphere, meshCone);
			delete[] h_gridPosFloat3;
		}

		void copyResultToCPUFromFloat3()
		{
            std::vector<float3> h_gridPosFloat3(m_nNodes);
			cutilSafeCall(cudaMemcpy(h_gridPosFloat3.data(), d_gridPosFloat3, sizeof(float3)*m_nNodes, cudaMemcpyDeviceToHost));

			for (SimpleMesh::VertexIter v_it = m_result.vertices_begin(); v_it != m_result.vertices_end(); ++v_it)
			{
				VertexHandle vh(v_it);

				int3   voxelId = m_vertexToVoxels[vh.idx()];
				float3 relativeCoords = m_relativeCoords[vh.idx()];

				float3 p000 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 0))];
				float3 p001 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 1))];
				float3 p010 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 0))];
				float3 p011 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 1))];
				float3 p100 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 0))];
				float3 p101 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 1))];
				float3 p110 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 0))];
				float3 p111 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 1))];

				float3 px00 = (1.0f - relativeCoords.x)*p000 + relativeCoords.x*p100;
				float3 px01 = (1.0f - relativeCoords.x)*p001 + relativeCoords.x*p101;
				float3 px10 = (1.0f - relativeCoords.x)*p010 + relativeCoords.x*p110;
				float3 px11 = (1.0f - relativeCoords.x)*p011 + relativeCoords.x*p111;

				float3 pxx0 = (1.0f - relativeCoords.y)*px00 + relativeCoords.y*px10;
				float3 pxx1 = (1.0f - relativeCoords.y)*px01 + relativeCoords.y*px11;

				float3 p = (1.0f - relativeCoords.z)*pxx0 + relativeCoords.z*pxx1;

				m_result.set_point(vh, SimpleMesh::Point(p.x, p.y, p.z));
			}
		}

	private:

		SimpleMesh m_result;
		SimpleMesh m_initial;

		int	   m_nNodes;

		float3 m_min;
		float3 m_max;

		int3   m_dims;
		float3 m_delta;

		int3*   m_vertexToVoxels;
		float3* m_relativeCoords;

		float3* d_gridPosTargetFloat3;
		float3* d_gridPosFloat3;
		float3* d_gridPosFloat3Urshape;
		float3* d_gridAnglesFloat3;

        CombinedSolverParameters m_params;

        std::unique_ptr<SolverBase>	m_warpingSolver;
        std::unique_ptr<SolverBase> m_warpingSolverCeres;
        std::unique_ptr<SolverBase>	m_warpingSolverOpt;
        std::unique_ptr<SolverBase>	m_warpingSolverOptLM;

        std::vector<SolverIteration> m_ceresIters;
        std::vector<SolverIteration> m_optIters;
        std::vector<SolverIteration> m_optLMIters;
};

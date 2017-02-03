#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

int main(int argc, const char * argv[])
{
	std::string filename = "meshes/neptune.ply";
    //std::string filename = "meshes/long.ply";
	if (argc >= 2) {
		filename = argv[1];
	}

	SimpleMesh* mesh = new SimpleMesh();

	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}
    printf("Faces: %d\nVertices: %d\n", mesh->n_faces(), mesh->n_vertices());

    CombinedSolverParameters params;
    params.optDoublePrecision = true;
    params.profileSolve = true;
    params.useCUDA = true;
    params.useOptLM = true;
    params.useCeres = true;
    params.nonLinearIter = 20;
    params.linearIter = 60;

    CombinedSolver solver(mesh, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    solver.saveGraphResults();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}

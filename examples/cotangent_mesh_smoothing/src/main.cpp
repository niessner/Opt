#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

int main(int argc, const char * argv[])
{
    // Bunny
    //std::string filename = "bunny.off";

    std::string filename = "serapis.stl";
    if (argc >= 2) {
        filename = argv[1];
    }
    bool performanceRun = false;
    if (argc >= 3) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
        }
        else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }


    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    CombinedSolverParameters params;
    params.useOpt = true;
    params.useOptLM = true;
    params.nonLinearIter = 8;
    params.linearIter = 25;
    CombinedSolver solver(mesh, performanceRun, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    if (!OpenMesh::IO::write_mesh(*res, "out.off"))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << "out.off" << std::endl;
        exit(1);
    }

	return 0;
}

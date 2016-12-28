#include "main.h"
#include "ImageWarping.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"

int main(int argc, const char * argv[])
{
	std::string filename = "Armadillo20k.ply";
    const char* markerFilename = "armadillo.mrk";
    //std::string filename = "raptor_clean.stl";
    //const char* markerFilename = "raptor.mrk";

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

	// Load Constraints
	LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename);

	std::vector<int>				constraintsIdx;
	std::vector<std::vector<float>> constraintsTarget;

	for (unsigned int i = 0; i < markersMesh.size(); i++)
	{
		constraintsIdx.push_back(markersMesh[i].getVertexIndex());
		constraintsTarget.push_back(markersMesh[i].getPosition());
	}

	SimpleMesh* mesh = new SimpleMesh();

	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}
    printf("Faces: %d\nVertices: %d\n", mesh->n_faces(), mesh->n_vertices());

	ImageWarping warping(mesh, constraintsIdx, constraintsTarget, performanceRun);
    SimpleMesh* res = warping.solve();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}

#include "main.h"
#include "ImageWarping.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>
int main(int argc, const char * argv[])
{
	//std::string filename = "Armadillo20k.ply";
    //const char* markerFilename = "armadillo.mrk";
    //std::string filename = "raptor_clean.stl";
    //const char* markerFilename = "raptor.mrk";
	std::string filename = "small_armadillo.ply";
	const char* markerFilename = "small_armadillo.mrk";


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
	int subdivisionFactor = 0;
	bool lmOnlyFullSolve = false;
	if (argc > 3) {
		lmOnlyFullSolve = true;
		subdivisionFactor = atoi(argv[3]);
		markerFilename = "small_armadillo.mrk";
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

	OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
	// Initialize subdivider
	if (lmOnlyFullSolve) {
		if (subdivisionFactor > 0) {
			subdivider.attach(*mesh);
			subdivider(subdivisionFactor);
			subdivider.detach();
		}
	} else {
		//OpenMesh::Subdivider::Uniform::CatmullClarkT<SimpleMesh> catmull;
		// Execute 1 subdivision steps
		subdivider.attach(*mesh);
		subdivider(1);
		subdivider.detach();
	}
	
	

    printf("Faces: %d\nVertices: %d\n", mesh->n_faces(), mesh->n_vertices());

	ImageWarping warping(mesh, constraintsIdx, constraintsTarget, performanceRun, lmOnlyFullSolve);
    SimpleMesh* res = warping.solve();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}

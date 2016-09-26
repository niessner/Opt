#include "main.h"
#include "ImageWarping.h"
#include "OpenMesh.h"

static SimpleMesh* createMesh(std::string filename) {
    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", mesh->n_faces(), mesh->n_vertices());
    return mesh;
}

int main(int argc, const char * argv[])
{
	std::string source_filename = "Armadillo20k.ply";
    std::string target_filename = "Armadillo20k.ply";

	if (argc >= 2) {
        source_filename = argv[1];
	}
    if (argc >= 3) {
        target_filename = argv[2];
    }

    SimpleMesh* sourceMesh = createMesh(source_filename);
    SimpleMesh* targetMesh = createMesh(target_filename);

    ImageWarping warping(sourceMesh, targetMesh);
    SimpleMesh* res = warping.solve();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

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
    std::string sourceDirectory = "handstand";
    std::vector<std::string> allFiles = ml::Directory::enumerateFiles(sourceDirectory);
    std::string source_filename = sourceDirectory + "/" + allFiles[0];

    SimpleMesh* sourceMesh = createMesh(source_filename);
    std::vector<SimpleMesh*> targetMeshes;
    for (int i = 1; i < allFiles.size(); ++i) {
        targetMeshes.push_back(createMesh(sourceDirectory + "/" + allFiles[i]));
    }
    std::cout << "All meshes now in memory" << std::endl;
    ImageWarping warping(sourceMesh, targetMeshes);
    SimpleMesh* res = warping.solve();
    
	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    
    for (SimpleMesh* mesh : targetMeshes) {
        delete mesh;
    }
    delete sourceMesh;

#ifdef _WIN32
	getchar();
#endif
	return 0;
}

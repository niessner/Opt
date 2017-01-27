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

static std::vector<int4> getSourceTetIndices(std::string filename) {
    // TODO: error handling
    std::ifstream inFile(filename);
    int tetCount = 0;
    int temp;
    inFile >> tetCount >> temp >> temp;
    std::vector<int4> tets(tetCount);
    for (int i = 0; i < tetCount; ++i) {
        inFile >> temp >> tets[i].x >> tets[i].y >> tets[i].z >> tets[i].w;
    }
    int4 f = tets[tets.size() - 1];
    printf("Final tet read: %d %d %d %d\n", f.x, f.y, f.z, f.w);
    return tets;
}

int main(int argc, const char * argv[])
{
    int startIndex = 41;
    std::string sourceDirectory = "squat";
    std::vector<std::string> allFiles = ml::Directory::enumerateFiles(sourceDirectory);
    std::string source_filename = sourceDirectory + "/" + allFiles[startIndex];


    std::vector<int4> sourceTetIndices = getSourceTetIndices(sourceDirectory + "_tet/" + "mesh_0000.1.ele");

    SimpleMesh* sourceMesh = createMesh(source_filename);

    std::vector<SimpleMesh*> targetMeshes;
    for (int i = startIndex + 1; i < allFiles.size(); i += 1) {
        targetMeshes.push_back(createMesh(sourceDirectory + "/" + allFiles[i]));
		if (targetMeshes.size() > 1) {
			break;
		}
    }
    std::cout << "All meshes now in memory" << std::endl;
    ImageWarping warping(sourceMesh, targetMeshes, sourceTetIndices, startIndex);
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

	return 0;
}

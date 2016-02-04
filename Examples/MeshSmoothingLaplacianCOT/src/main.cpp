#include "main.h"
#include "ImageWarping.h"
#include "OpenMesh.h"

int main(int argc, const char * argv[])
{
  // Bunny
  //std::string filename = "bunny.off";

  std::string filename = "serapis.stl";
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

	ImageWarping warping(mesh);
	SimpleMesh* res = warping.solve();

	if (!OpenMesh::IO::write_mesh(*res, "out.off"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	//	getchar();
	return 0;
}

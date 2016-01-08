#include "main.h"
#include "ImageSolver.h"
#include "SFSSolverInput.h"

int main(int argc, const char * argv[])
{

	std::string inputFilenamePrefix = "default";
	if (argc >= 2) {
        inputFilenamePrefix = std::string(argv[1]);
	}
    SFSSolverInput solverInput;
    
    solverInput.load(inputFilenamePrefix);
    solverInput.parameters.nNonLinearIterations = 6;
    //solverInput.parameters.nLinIterations = 100;
	
    ImageSolver solver(solverInput);
	printf("Solving\n");
    std::shared_ptr<SimpleBuffer> result = solver.solve();
	printf("Solved\n");
	printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
	printf("Save\n");
	#ifdef _WIN32
	getchar();
	#endif
	return 0;
}

#include "main.h"
#include "ImageSolver.h"
#include "SolverInput.h"

int main(int argc, const char * argv[])
{

	std::string inputFilenamePrefix = "default";
	if (argc >= 2) {
        inputFilenamePrefix = std::string(argv[1]);
	}
    SolverInput solverInput;
    solverInput.load(inputFilenamePrefix);
	
    ImageSolver solver(solverInput);
	printf("Solving\n");
    shared_ptr<SimpleBuffer> result = solver.solve();
	printf("Solved\n");
	printf("About to save\n");
    result->save("sfsOutput.imagedump");
	printf("Save\n");
	#ifdef _WIN32
	getchar();
	#endif
	return 0;
}

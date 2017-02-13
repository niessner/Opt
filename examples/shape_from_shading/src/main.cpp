#include "main.h"
#include "CombinedSolver.h"
#include "SFSSolverInput.h"

int main(int argc, const char * argv[])
{

    std::string inputFilenamePrefix = "default";
    if (argc >= 2) {
        inputFilenamePrefix = std::string(argv[1]);
    }

    bool performanceRun = false;
    if (argc > 2) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
        }
        else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }

    SFSSolverInput solverInputCPU, solverInputGPU;

    solverInputGPU.load(inputFilenamePrefix, true);

    solverInputGPU.parameters.nNonLinearIterations = 34;
    solverInputGPU.parameters.nLinIterations = 10;

	if (performanceRun) {
		solverInputGPU.parameters.nNonLinearIterations = 60;
		solverInputGPU.parameters.nLinIterations = 10;
	}

    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");
    solverInputCPU.load(inputFilenamePrefix, false);

    CombinedSolverParameters params;
    performanceRun = true;
    if (performanceRun) {
        params.useCUDA  = true;
        params.useOpt   = true;
        params.useOptLM = false;
        params.useCeres = false;
    }
    else {
        //m_params.useCUDA = true;
    }

    CombinedSolver solver(solverInputGPU, params);
    printf("Solving\n");
    solver.solveAll();
    std::shared_ptr<SimpleBuffer> result = solver.result();
    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    printf("Save\n");

	return 0;
}

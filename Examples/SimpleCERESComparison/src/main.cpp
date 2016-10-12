
#include "CombinedSolver.h"
#include <random>

int main(int argc, const char * argv[]) {
    int N = 16;
    double2 generatorParams = { 131.0, 83.1 };
    std::vector<double2> dataPoints(N);
    double a = generatorParams.x;
    double b = generatorParams.y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    for (int i = 0; i < dataPoints.size(); ++i) {
        double x = double(i)*2.0*3.141592653589 / N;
        double y = (a*cos(b*x) + b*sin(a*x));
        // Add in noise
        y += dis(gen);
        dataPoints[i].x = x;
        dataPoints[i].y = y;

    }
    // Generate data
    CombinedSolver solver(generatorParams, dataPoints);
    solver.solve();


    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

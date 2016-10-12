
#include "CombinedSolver.h"
#include <random>
#include <iostream>
int main(int argc, const char * argv[]) {
    int N = 512;
    double2 generatorParams = { 100.0, 102.0 };
    std::vector<double2> dataPoints(N);
    double a = generatorParams.x;
    double b = generatorParams.y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    for (int i = 0; i < dataPoints.size(); ++i) {
        double x = double(i)*2.0*3.141592653589 / N;
        double y = (a*cos(b*x) + b*sin(a*x));
        //y = a*x + b;
        // Add in noise
        //y += dis(gen);
        dataPoints[i].x = x;
        dataPoints[i].y = y;

    }
    double2 initalGuess = { 99.5, 102.5 };
    //initalGuess = generatorParams;
 
    CombinedSolver solver(initalGuess, dataPoints);
    double2 finalResult = solver.solve();
    std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;

    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

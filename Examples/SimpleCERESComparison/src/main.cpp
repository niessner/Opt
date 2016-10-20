
#include "CombinedSolver.h"
#include <random>
#include <iostream>

#include "util.h"

#ifdef _WIN32
// for getcwd
#include <Windows.h>
#endif

using namespace std;

vector<double2> loadFile(const string &filename)
{
	vector<double2> result;
	auto lines = Utility::GetFileLines(filename, 2);
	for (string line : lines)
	{
		double2 d;
		d.y = Utility::StringToFloat(Utility::PartitionString(line, " ")[0]);
		d.x = Utility::StringToFloat(Utility::PartitionString(line, " ")[1]);
		result.push_back(d);
	}
	return result;
}

void runTestA()
{
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
	UNKNOWNS initalGuess = { 99.5, 102.5, 0.0 };
	//initalGuess = generatorParams;

	CombinedSolver solver(initalGuess, dataPoints);
	UNKNOWNS finalResult = solver.solve();
	std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;
}

void runTestB()
{
	string problemDataFilename = "none";
	if (useProblemMisra) problemDataFilename = "misra.txt";
	if (useProblemBennet5) problemDataFilename = "bennet5.txt";

	auto dataPoints = loadFile("data/" + problemDataFilename);

	UNKNOWNS initialGuess = { 0.0, 0.0, 0.0 };
	if (useProblemMisra) initialGuess = { 500.0, 1e-4, 0.0 };
	if (useProblemBennet5) initialGuess = {-2.0e3, 50.0, 0.8};

	if (initialGuess.x == 0.0)
	{
		cout << "problem not specified" << endl;
		cin.get();
	}

	CombinedSolver solver(initialGuess, dataPoints);
	UNKNOWNS finalResult = solver.solve();
	//std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;


	cout << "Ceres solution: " << solver.m_ceresResult.x << " " << solver.m_ceresResult.y << " " << solver.m_ceresResult.z << endl;
	cout << "Opt solution: " << solver.m_optResult.x << " " << solver.m_optResult.y << " " << solver.m_optResult.z << endl;
	
	if (useProblemMisra) cout << "True solution: " << misraSolution.x << " " << misraSolution.y << " " << misraSolution.z << endl;
	if (useProblemMisra) cout << "True solution: " << bennet5Solution.x << " " << bennet5Solution.y << " " << bennet5Solution.z << endl;
}

int main(int argc, const char * argv[]) {
#ifdef _WIN32
	char cwd[1000];
	GetCurrentDirectoryA(1000, cwd);
	//cout << "current dir: " << cwd << endl;
#endif

	if (useProblemDefault)
	{
		runTestA();
	}
	else
	{
		runTestB();
	}

    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

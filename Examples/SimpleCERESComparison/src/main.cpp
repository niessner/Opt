
#include "CombinedSolver.h"
#include <random>
#include <iostream>

#include "util.h"

#ifdef _WIN32
// for getcwd
#include <Windows.h>
#endif

using namespace std;

vector<NLLSProblem> makeProblems()
{
	vector<NLLSProblem> problems;

	//problems.push_back(NLLSProblem("misra", 2, { 2.3894212918E+02, 5.5015643181E-04, 0.0 }, { 2.3894212918E+02, 5.5015643181E-04, 0.0 }));
	problems.push_back(NLLSProblem("bennet5", 2, { -200, 50.0, 0.8 }, { -2.5235e3, 4.6736e1, 9.32184e-1 }));
	//problems.push_back(NLLSProblem("chwirut1", 2, { 1e-1, 1e-2, 2e-2 }, { 1.9027818370E-01, 6.1314004477E-03, 1.0530908399E-02 }));

	return problems;
}

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
	
	NLLSProblem problem("curveFitting", 2, { 99.5, 102.5, 0.0 }, { 100.0, 102.0, 0.0 });
	
	CombinedSolver solver(problem, dataPoints);
	UNKNOWNS finalResult = solver.solve(problem);
	std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;
}

ofstream resultFile("results.txt");
void runProblem(const NLLSProblem &problem)
{
	auto dataPoints = loadFile("data/" + problem.baseName + ".txt");

	CombinedSolver solver(problem, dataPoints);
	UNKNOWNS finalResult = solver.solve(problem);
	//std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;

	resultFile << "Problem: " << problem.baseName << endl;
	resultFile << "True solution: " << problem.trueSolution.x << " " << problem.trueSolution.y << " " << problem.trueSolution.z << endl;
	resultFile << "Ceres solution: " << solver.m_ceresResult.x << " " << solver.m_ceresResult.y << " " << solver.m_ceresResult.z << endl;
	resultFile << "Opt solution: " << solver.m_optResult.x << " " << solver.m_optResult.y << " " << solver.m_optResult.z << endl << endl;
}

void runTestB()
{
	for (auto &p : makeProblems())
	{
		runProblem(p);
	}
	
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

	cout << "See results.txt" << endl;
    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

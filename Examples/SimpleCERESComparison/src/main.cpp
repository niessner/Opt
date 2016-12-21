
#include "CombinedSolver.h"
#include <random>
#include <iostream>
#include <iomanip>
#include "util.h"

#ifdef _WIN32
// for getcwd
#include <Windows.h>
#include <direct.h>
#endif

//http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

using namespace std;

vector<NLLSProblem> makeProblems()
{
	vector<NLLSProblem> problems;

	//problems.push_back(NLLSProblem("mgh09", 4, { 25.0, 39.0, 41.5, 39.0 }, { 1.9280693458E-01, 1.9128232873E-01, 1.2305650693E-01, 1.3606233068E-01 }));
    //problems.push_back(NLLSProblem("eckerle4", 3, { 1.0, 10.0, 500.0, 0.0 }, { 1.5543827178E+00, 4.0888321754E+00, 4.5154121844E+02, 0.0 }));
	//problems.push_back(NLLSProblem("rat42", 3, { 100.0, 1.0, 0.1, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }));

	//problems.push_back(NLLSProblem("misra", 2, { 5e2, 1e-4, 0.0, 0.0 }, { 2.3894212918E+02, 5.5015643181E-04, 0.0, 0.0 }));
	//problems.push_back(NLLSProblem("bennet5", 2, { -200, 50.0, 0.8, 0.0 }, { -2.5235e3, 4.6736e1, 9.32184e-1, 0.0 }));
	problems.push_back(NLLSProblem("chwirut1", 2, { 1e-1, 1e-2, 2e-2, 0.0 }, { 1.9027818370E-01, 6.1314004477E-03, 1.0530908399E-02, 0.0 }));
	
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
    std::cout << "Final Result: " << finalResult.vals[0] << ", " << finalResult.vals[1] << std::endl;
}

template<class T>
const T& clampedRead(const std::vector<T> &v, int index)
{
	if (index < 0) return v[0];
	if (index >= v.size()) return v[v.size() - 1];
	return v[index];
}


void runProblem(const NLLSProblem &problem)
{
    auto dataPoints = loadFile("data/" + problem.baseName + ".txt");
	CombinedSolver solver(problem, dataPoints);
	UNKNOWNS finalResult = solver.solve(problem);

    
    std::string resultDirectory = "results/";
#   if OPT_DOUBLE_PRECISION
    std::string resultSuffix = "_double";
#   else
    std::string resultSuffix = "_float";
#   endif

    ofstream resultFile(resultDirectory + problem.baseName + resultSuffix + ".csv");
	//std::cout << "Final Result: " << finalResult.x << ", " << finalResult.y << std::endl;
    resultFile << std::scientific;
    resultFile << std::setprecision(20);
	resultFile << "Problem, " << problem.baseName << endl;

    auto printSolution = [](ofstream& file, std::string name, unsigned unknownCount, double9 solution){
        file << name << " solution, ";
        for (int i = 0; i < unknownCount; ++i) {
            file << ", " << solution.vals[i];
        }
        file << endl;
    };

    printSolution(resultFile, "True",  problem.unknownCount, problem.trueSolution);
    printSolution(resultFile, "Ceres", problem.unknownCount, solver.m_ceresResult);
    printSolution(resultFile, "Opt",   problem.unknownCount, solver.m_optResult);

	resultFile << "Iter, Ceres Error, Opt Error, Ceres Iter Time (ms), Opt Iter Time (ms), Total Ceres Time (ms), Total Opt Time (ms)" << endl;
    double sumOptTime = 0.0;
    double sumCeresTime = 0.0;
	for (int i = 0; i < (int)max(solver.m_ceresIters.size(), solver.m_optIters.size()); i++)
	{
        double ceresTime    = ((solver.m_ceresIters.size() > i) ? solver.m_ceresIters[i].timeInMS : 0.0);
        double optTime = ((solver.m_optIters.size() > i) ? solver.m_optIters[i].timeInMS : 0.0);
        sumCeresTime    += ceresTime;
        sumOptTime      += optTime;
        resultFile << i << ", " << clampedRead(solver.m_ceresIters, i).cost << ", " << clampedRead(solver.m_optIters, i).cost << ", " << ceresTime << ", " << optTime << ", " << sumCeresTime << ", " << sumOptTime << endl;
	}
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

	cout << "See results/*.csv" << endl;
    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

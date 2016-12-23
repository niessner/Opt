
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

// http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

using namespace std;

vector<NLLSProblem> makeProblems()
{
	vector<NLLSProblem> problems;

	problems.push_back(NLLSProblem("bennett5", 3, { -2000.0, 50.0, 0.8 }, { -2523.5058043, 46.736564644, 0.93218483193}));
	problems.push_back(NLLSProblem("boxbod", 2, { 1.0, 1.0 }, { 213.80940889, 0.54723748542}));
	problems.push_back(NLLSProblem("chwirut1", 3, { 0.1, 0.01, 0.02 }, { 0.1902781837, 0.0061314004477, 0.010530908399}));
	problems.push_back(NLLSProblem("chwirut2", 3, { 0.1, 0.01, 0.02 }, { 0.16657666537, 0.0051653291286, 0.012150007096}));
	problems.push_back(NLLSProblem("danwood", 2, { 1.0, 5.0 }, { 0.76886226176, 3.8604055871}));
	problems.push_back(NLLSProblem("eckerle4", 3, { 1.0, 10.0, 500.0 }, { 1.5543827178, 4.0888321754, 451.54121844}));
	problems.push_back(NLLSProblem("enso", 9, { 11.0, 3.0, 0.5, 40.0, -0.7, -1.3, 25.0, -0.3, 1.4 }, { 10.510749193, 3.0762128085, 0.53280138227, 44.3110887, -1.6231428586, 0.52554493756, 26.88761444, 0.21232288488, 1.4966870418}));
	problems.push_back(NLLSProblem("gauss1", 8, { 97.0, 0.009, 100.0, 65.0, 20.0, 70.0, 178.0, 16.5 }, { 98.778210871, 0.010497276517, 100.48990633, 67.481111276, 23.12977336, 71.994503004, 178.99805021, 18.389389025}));
	problems.push_back(NLLSProblem("gauss2", 8, { 96.0, 0.009, 103.0, 106.0, 18.0, 72.0, 151.0, 18.0 }, { 99.018328406, 0.010994945399, 101.88022528, 107.03095519, 23.578584029, 72.045589471, 153.27010194, 19.525972636}));
	problems.push_back(NLLSProblem("gauss3", 8, { 94.9, 0.009, 90.1, 113.0, 20.0, 73.8, 140.0, 20.0 }, { 98.94036897, 0.010945879335, 100.69553078, 111.63619459, 23.300500029, 73.705031418, 147.76164251, 19.66822123}));
	problems.push_back(NLLSProblem("hahn1", 7, { 10.0, -1.0, 0.05, -1e-05, -0.05, 0.001, -1e-06 }, { 1.0776351733, -0.12269296921, 0.004086375061, -1.4262662514e-06, -0.0057609940901, 0.00024053735503, -1.2314450199e-07}));
	problems.push_back(NLLSProblem("kirby2", 5, { 2.0, -0.1, 0.003, -0.001, 1e-05 }, { 1.6745063063, -0.13927397867, 0.0025961181191, -0.001724181187, 2.1664802578e-05}));
	problems.push_back(NLLSProblem("lanczos1", 6, { 1.2, 0.3, 5.6, 5.5, 6.5, 7.6 }, { 0.095100000027, 1.0000000001, 0.86070000013, 3.0000000002, 1.5575999998, 5.0000000001}));
	problems.push_back(NLLSProblem("lanczos2", 6, { 1.2, 0.3, 5.6, 5.5, 6.5, 7.6 }, { 0.096251029939, 1.0057332849, 0.86424689056, 3.0078283915, 1.5529016879, 5.00287981}));
	problems.push_back(NLLSProblem("lanczos3", 6, { 1.2, 0.3, 5.6, 5.5, 6.5, 7.6 }, { 0.086816414977, 0.95498101505, 0.84400777463, 2.9515951832, 1.5825685901, 4.9863565084}));
	problems.push_back(NLLSProblem("mgh09", 4, { 25.0, 39.0, 41.5, 39.0 }, { 0.19280693458, 0.19128232873, 0.12305650693, 0.13606233068}));
    problems.push_back(NLLSProblem("mgh10", 3, { 2.0, 400000.0, 25000.0 }, { 0.005609636471, 6181.3463463, 345.22363462 }));
	problems.push_back(NLLSProblem("mgh17", 5, { 50.0, 150.0, -100.0, 1.0, 2.0 }, { 0.37541005211, 1.9358469127, -1.4646871366, 0.01286753464, 0.022122699662}));
	problems.push_back(NLLSProblem("misra1a", 2, { 500.0, 0.0001 }, { 238.94212918, 0.00055015643181}));
	problems.push_back(NLLSProblem("misra1b", 2, { 500.0, 0.0001 }, { 337.99746163, 0.00039039091287}));
	problems.push_back(NLLSProblem("misra1c", 2, { 500.0, 0.0001 }, { 636.42725809, 0.00020813627256}));
	problems.push_back(NLLSProblem("misra1d", 2, { 500.0, 0.0001 }, { 437.36970754, 0.00030227324449}));
	//problems.push_back(NLLSProblem("nelson", 3, { 2.0, 0.0001, -0.01 }, { 2.5906836021, 5.6177717026e-09, -0.057701013174}));
	problems.push_back(NLLSProblem("rat42", 3, { 100.0, 1.0, 0.1 }, { 72.462237576, 2.6180768402, 0.067359200066}));
	problems.push_back(NLLSProblem("rat43", 4, { 100.0, 10.0, 1.0, 1.0 }, { 699.6415127, 5.2771253025, 0.75962938329, 1.2792483859}));
	problems.push_back(NLLSProblem("roszman1", 4, { 0.1, -1e-05, 1000.0, -100.0 }, { 0.20196866396, -6.1953516256e-06, 1204.4556708, -181.34269537}));
	problems.push_back(NLLSProblem("thurber", 7, { 1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03 }, { 1288.13968, 1491.0792535, 583.23836877, 75.416644291, 0.96629502864, 0.39797285797, 0.049727297349}));

    /* The problematic ones 
    problems.push_back(NLLSProblem("mgh10", 3, { 2.0, 400000.0, 25000.0 }, { 0.005609636471, 6181.3463463, 345.22363462 }));
    problems.push_back(NLLSProblem("mgh17", 5, { 50.0, 150.0, -100.0, 1.0, 2.0 }, { 0.37541005211, 1.9358469127, -1.4646871366, 0.01286753464, 0.022122699662 }));
    */

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

    printSolution(resultFile, "True", problem.unknownCount, problem.trueSolution);
    printSolution(resultFile, "Ceres", problem.unknownCount, solver.m_ceresResult);
    printSolution(resultFile, "Opt", problem.unknownCount, solver.m_optResult);

    resultFile << "Iter, Ceres Error, Opt Error, Ceres Iter Time (ms), Opt Iter Time (ms), Total Ceres Time (ms), Total Opt Time (ms)" << endl;
    double sumOptTime = 0.0;
    double sumCeresTime = 0.0;
    if (solver.m_ceresIters.size() == 0) {
        solver.m_ceresIters.push_back({ 0.0, 0.0 });
    }
    double prevCeresCost = std::numeric_limits<double>::infinity();
	for (int i = 0; i < (int)max(solver.m_ceresIters.size(), solver.m_optIters.size()); i++)
	{
        double ceresTime    = ((solver.m_ceresIters.size() > i) ? solver.m_ceresIters[i].timeInMS : 0.0);
        double optTime = ((solver.m_optIters.size() > i) ? solver.m_optIters[i].timeInMS : 0.0);
        sumCeresTime    += ceresTime;
        sumOptTime      += optTime;

        // When hooked up to certain debug builds of CERES, the "potential" cost is reported instead of the real one. This corrects for that.
        double ceresCost = clampedRead(solver.m_ceresIters, i).cost;
        ceresCost = fmin(ceresCost, prevCeresCost);
        prevCeresCost = ceresCost;

        resultFile << i << ", " << ceresCost << ", " << clampedRead(solver.m_optIters, i).cost << ", " << ceresTime << ", " << optTime << ", " << sumCeresTime << ", " << sumOptTime << endl;
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

#pragma once
#include <limits>
#include <algorithm>


struct SolverIteration
{
    SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
    double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};

template<class T>
const T& clampedRead(const std::vector<T> &v, int index)
{
    if (index < 0) return v[0];
    if (index >= v.size()) return v[v.size() - 1];
    return v[index];
}

static void saveSolverResults(std::string directory, std::string suffix,
    std::vector<SolverIteration>& ceresIters, std::vector<SolverIteration>& optGNIters, std::vector<SolverIteration>& optLMIters) {
    std::ofstream resultFile(directory + "results" + suffix + ".csv");
    resultFile << std::scientific;
    resultFile << std::setprecision(20);

    resultFile << "Iter, Ceres Error, Opt (GN) Error,  Opt (LM) Error, Ceres Iter Time (ms), Opt (GN) Iter Time (ms), Opt (LM) Iter Time (ms), Total Ceres Time (ms), Total Opt (GN) Time (ms), Total Opt (LM) Time (ms)" << std::endl;
    double sumOptGNTime = 0.0;
    double sumOptLMTime = 0.0;
    double sumCeresTime = 0.0;
    if (ceresIters.size() == 0) {
        ceresIters.push_back(SolverIteration(0, 0));
    }
    if (optLMIters.size() == 0) {
        optLMIters.push_back(SolverIteration(0, 0));
    }
    if (optGNIters.size() == 0) {
        optGNIters.push_back(SolverIteration(0, 0));
    }
    for (int i = 0; i < (int)std::max((int)ceresIters.size(), std::max((int)optLMIters.size(), (int)optGNIters.size())); i++)
    {
        double ceresTime = ((ceresIters.size() > i) ? ceresIters[i].timeInMS : 0.0);
        double optGNTime = ((optGNIters.size() > i) ? optGNIters[i].timeInMS : 0.0);
        double optLMTime = ((optLMIters.size() > i) ? optLMIters[i].timeInMS : 0.0);
        sumCeresTime += ceresTime;
        sumOptGNTime += optGNTime;
        sumOptLMTime += optLMTime;
        resultFile << i << ", " << clampedRead(ceresIters, i).cost << ", " << clampedRead(optGNIters, i).cost << ", " << clampedRead(optLMIters, i).cost << ", " << ceresTime << ", " << optGNTime << ", " << optLMTime << ", " << sumCeresTime << ", " << sumOptGNTime << ", " << sumOptLMTime << std::endl;
    }
}
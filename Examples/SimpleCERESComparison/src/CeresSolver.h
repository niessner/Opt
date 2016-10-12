#pragma once
#include <vector>
class CeresSolver {
public:
	CeresSolver(int N)
	{
        functionData.resize(N);
	}

    void CeresSolver::solve(
        double2* funcParameters,
        double2* funcData);

private:
    double2 functionParameters;
    std::vector<double2> functionData;
};


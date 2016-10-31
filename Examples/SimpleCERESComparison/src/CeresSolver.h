#pragma once
#include <vector>
class CeresSolver {
public:
	CeresSolver(size_t N)
	{
        functionData.resize(N);
	}

	std::vector<SolverIteration> CeresSolver::solve(
		const NLLSProblem &problem,
        UNKNOWNS* funcParameters,
        double2* funcData);

private:
    std::vector<double2> functionData;
};


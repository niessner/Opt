#pragma once
#include "SolverIteration.h"
#include "NamedParameters.h"
#include "SolverPerformanceSummary.h"

class SolverBase {
public:
    SolverBase() {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, SolverPerformanceSummary& perfStats, bool profileSolve, std::vector<SolverIteration>& iter) {
        fprintf(stderr, "No solve implemented\n");
        return m_finalCost;
    }
    double finalCost() const {
        return m_finalCost;
    }
    SolverPerformanceSummary getSummaryStatistics() const {
        return m_summaryStats;
    }
protected:
    double m_finalCost = nan("");
    SolverPerformanceSummary m_summaryStats = {};
};
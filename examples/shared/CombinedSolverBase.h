#pragma once


#include "OptSolver.h"
#include "CombinedSolverParameters.h"
#include "SolverIteration.h"


/** We want to run several solvers in an identical manner, with some initalization
and finish code for each of the examples. The structure is the same for every
example, so we keep it in solveAll(), and let individual examples override
combinedSolveInit(); combinedSolveFinalize(); preSingleSolve(); postSingleSolve();*/
class CombinedSolverBase {
public:
    virtual void combinedSolveInit() = 0;
    virtual void combinedSolveFinalize() = 0;
    virtual void preSingleSolve() = 0;
    virtual void postSingleSolve() = 0;
    virtual void preNonlinearSolve(int iteration) = 0;
    virtual void postNonlinearSolve(int iteration) = 0;

    void solveAll() {
        combinedSolveInit();
        for (auto s : m_solverInfo) {
            if (s.enabled) {
                preSingleSolve();
                if (m_combinedSolverParameters.numIter == 1) {
                    preNonlinearSolve(0);
                    std::cout << "//////////// (" << s.name << ") ///////////////" << std::endl;
                    s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s.iterationInfo);
                    postNonlinearSolve(0);
                } else {
                    for (int i = 0; i < m_combinedSolverParameters.numIter; ++i) {
                        std::cout << "//////////// ITERATION" << i << "  (" << s.name << ") ///////////////" << std::endl;
                        preNonlinearSolve(i);
                        s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s.iterationInfo);
                        postNonlinearSolve(i);
                        if (m_combinedSolverParameters.earlyOut) {
                            break;
                        }
                    }
                }
                postSingleSolve();
            }
        }
        combinedSolveFinalize();
    }

    double getCost(std::string name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->finalCost();
                }
            }
        }
        return nan(nullptr);
    }

    std::vector<SolverIteration> getIterationInfo(std::string name) {
        for (auto& s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.iterationInfo;
                }
            }
        }
        return std::vector<SolverIteration>();
    }

    void ceresIterationComparison(std::string name) {
        saveSolverResults("results/", OPT_DOUBLE_PRECISION ? "_double" : "_float", getIterationInfo("Ceres"), getIterationInfo("Opt(GN)"), getIterationInfo("Opt(LM)"));
        reportFinalCosts(name, m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), getCost("Ceres"));
    }

    void addSolver(std::shared_ptr<SolverBase> solver, std::string name, bool enabled = true) {
        m_solverInfo.resize(m_solverInfo.size() + 1);
        m_solverInfo[m_solverInfo.size() - 1].set(solver, name, enabled);

    }

    void addOptSolvers(std::vector<unsigned int> dims, std::string problemFilename) {
        if (m_combinedSolverParameters.useOpt) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "gaussNewtonGPU"), "Opt(GN)", true);
        }
        if (m_combinedSolverParameters.useOptLM) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "LMGPU"), "Opt(LM)", true);
        }
    }



protected:
    struct SolverInfo {
        std::shared_ptr<SolverBase> solver;
        std::vector<SolverIteration> iterationInfo;
        std::string name;
        bool enabled;
        void set(std::shared_ptr<SolverBase> _solver, std::string _name, bool _enabled) {
            solver = std::move(_solver);
            name = _name;
            enabled = _enabled;
        }
    };
    std::vector<SolverInfo> m_solverInfo;

    NamedParameters m_solverParams;
    NamedParameters m_problemParams;
    CombinedSolverParameters m_combinedSolverParameters;
};
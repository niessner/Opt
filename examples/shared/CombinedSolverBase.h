#pragma once

#include "OptSolver.h"
#include "CombinedSolverParameters.h"
#include "SolverIteration.h"
#include "Config.h"

static std::string qt(std::string str) { return "\"" + str + "\""; }

static void toStream(std::string name, SolverPerformanceEntry& entry, std::ostream& out, std::string ident, bool commaAfter = true) {
    std::string ender = commaAfter ? "," : "";
    out << ident + qt(name) + " : {" << std::endl;
    std::string newIdent = ident + "  ";
    auto emit = [&](std::string fieldname, double field, std::string ending) {out << newIdent << qt(fieldname) + " : " << field << ending << std::endl; };
    out << newIdent << qt("count") + " : " << entry.count << "," << std::endl;
    emit("minMS", entry.minMS, ",");
    emit("maxMS", entry.maxMS, ",");
    emit("meanMS", entry.meanMS,",");
    emit("stddevMS", entry.stddevMS,"");
    out << ident << "}" << ender << std::endl;
}

static void toStream(SolverPerformanceSummary& summary, std::ostream& out, std::string ident) {
    std::string newIdent = ident + "  ";
    out << "{" << std::endl;
    toStream("total",               summary.total,              out, newIdent);
    toStream("nonlinearIteration",  summary.nonlinearIteration, out, newIdent);
    toStream("nonlinearSetup",      summary.nonlinearSetup,     out, newIdent);
    toStream("linearSolve",         summary.linearSolve,        out, newIdent);
    toStream("nonlinearResolve",    summary.nonlinearResolve,   out, newIdent, false);
    out << ident << "}";
}

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

    CombinedSolverBase(std::string name) : m_name(name) {}

    void reportFinalSolverCosts(std::string name, std::ostream& output = std::cout) {
        reportFinalCosts(name, m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), getCost("Ceres"), output);
    }

    void saveFinalCosts(std::string name) {
        std::ofstream ofs("finalCosts.json");
        if (ofs.good()) {
            reportFinalSolverCosts(name, ofs);
        } else {
            std::cout << "Error opening finalCosts.json" << std::endl;
        }
    }

    void reportPerformanceStatistics(std::string name, std::ostream& output = std::cout) {
        output << "{  \"name\" : \"" << name << "\"," << std::endl;
        output << "  \"performance\" : {" << std::endl;

        output << std::scientific;
        output << std::setprecision(18);

        std::vector<std::pair<std::string, SolverPerformanceSummary>> perf;

        if (m_combinedSolverParameters.useOpt)      perf.push_back({ "OptGN", getPerfStats("Opt(GN)") });
        if (m_combinedSolverParameters.useOptLM)    perf.push_back({ "OptLM", getPerfStats("Opt(LM)") });
        if (m_combinedSolverParameters.useCeres)    perf.push_back({ "Ceres", getPerfStats("Ceres") });
        if (m_combinedSolverParameters.useCUDA)     perf.push_back({ "Cuda", getPerfStats("CUDA") });

        for (int i = 0; i < perf.size(); ++i) {
            auto delim = (i != perf.size() - 1) ? "," : "";
            output << "    \"" << perf[i].first << "\" : ";
            toStream(perf[i].second, output, "    ");
            output << delim << std::endl;
        }
        output << "  }" << std::endl << "}" << std::endl;
    }

    void savePerformanceStatistics(std::string name) {
        std::ofstream ofs("perf.json");
        if (ofs.good()) {
            reportPerformanceStatistics(name, ofs);
        } else {
            std::cout << "Error opening perf.json" << std::endl;
        }
    }

    virtual void solveAll() {
        combinedSolveInit();
        for (auto& s : m_solverInfo) {
            if (s.enabled) {
                singleSolve(s);
            }
        }
        combinedSolveFinalize();
        if (m_combinedSolverParameters.profileSolve) {
            ceresIterationComparison(m_name, m_combinedSolverParameters.optDoublePrecision);
        }
        saveFinalCosts(m_name);
        savePerformanceStatistics(m_name);
    }

    double getCost(std::string name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->finalCost();
                }
            }
        }
        return nan("");
    }
    SolverPerformanceSummary getPerfStats(std::string name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->getSummaryStatistics();
                }
            }
        }
        return{};
    }

    void setParameters(const CombinedSolverParameters& params) {
        m_combinedSolverParameters = params;
        if (params.useCeres && !USE_CERES) {
            printf("Ceres not enabled in this build, turning off Ceres as an active solver.\n");
            m_combinedSolverParameters.useCeres = false;
        }
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

    void ceresIterationComparison(std::string name, bool optDoublePrecision) {
        saveSolverResults("results/", optDoublePrecision ? "_double" : "_float", getIterationInfo("Ceres"), getIterationInfo("Opt(GN)"), getIterationInfo("Opt(LM)"), optDoublePrecision);
    }

    void addSolver(std::shared_ptr<SolverBase> solver, std::string name, bool enabled = true) {
        m_solverInfo.resize(m_solverInfo.size() + 1);
        m_solverInfo[m_solverInfo.size() - 1].set(solver, name, enabled);

    }

    void addOptSolvers(std::vector<unsigned int> dims, std::string problemFilename, bool doublePrecision = false) {
        if (m_combinedSolverParameters.useOpt) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "gaussNewtonGPU", doublePrecision), "Opt(GN)", true);
        }
        if (m_combinedSolverParameters.useOptLM) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "LMGPU", doublePrecision), "Opt(LM)", true);
        }
    }

    std::string activeSolverName() const {
        return m_activeSolverInfo.name;
    }

protected:
    struct SolverInfo {
        std::shared_ptr<SolverBase> solver;
        std::vector<SolverIteration> iterationInfo;
        std::string name;
        bool enabled;
        SolverPerformanceSummary perfSummary;
        void set(std::shared_ptr<SolverBase> _solver, std::string _name, bool _enabled) {
            solver = std::move(_solver);
            name = _name;
            enabled = _enabled;
        }
    };
    std::vector<SolverInfo> m_solverInfo;

    virtual void singleSolve(SolverInfo& s) {
        m_activeSolverInfo = s;
        preSingleSolve();
        if (m_combinedSolverParameters.numIter == 1) {
            preNonlinearSolve(0);
            std::cout << "//////////// (" << s.name << ") ///////////////" << std::endl;
            s.solver->solve(m_solverParams, m_problemParams, s.perfSummary, m_combinedSolverParameters.profileSolve, s.iterationInfo);
            postNonlinearSolve(0);
        } else {
            for (int i = 0; i < (int)m_combinedSolverParameters.numIter; ++i) {
                std::cout << "//////////// ITERATION" << i << "  (" << s.name << ") ///////////////" << std::endl;
                preNonlinearSolve(i);
                s.solver->solve(m_solverParams, m_problemParams, s.perfSummary, m_combinedSolverParameters.profileSolve, s.iterationInfo);
                postNonlinearSolve(i);
                if (m_combinedSolverParameters.earlyOut || m_endSolveEarly) {
                    m_endSolveEarly = false;
                    break;
                }
            }
        }
        postSingleSolve();
    }
    SolverInfo m_activeSolverInfo;
    // Set to true in preNonlinearSolve or postNonlinearSolve to finish the solve before the specified number of iterations
    bool m_endSolveEarly = false;
    NamedParameters m_solverParams;
    NamedParameters m_problemParams;
    CombinedSolverParameters m_combinedSolverParameters;
    std::string m_name = "Default";
};
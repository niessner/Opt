#pragma once
extern "C" {
#include <Opt.h>
}
#include "SolverIteration.h"
#include <vector>
#include "cudaUtil.h"

#ifdef _WIN32
#include <Windows.h>
class SimpleTimer {
public:
    void init() {
        // get ticks per second
        QueryPerformanceFrequency(&frequency);

        // start timer
        QueryPerformanceCounter(&lastTick);
    }
    // Time since last tick in ms
    double tick() {
        LARGE_INTEGER currentTick;
        QueryPerformanceCounter(&currentTick);

        // compute and print the elapsed time in millisec
        double elapsedTime = (currentTick.QuadPart - lastTick.QuadPart) * 1000.0 / frequency.QuadPart;
        lastTick = currentTick;
        return elapsedTime;
    }
protected:
    LARGE_INTEGER frequency;
    LARGE_INTEGER lastTick;
};
#else
class SimpleTimer {
public:
    void init() {}
    // Time since last tick in ms
    double tick() { return fnan();}
};
#endif



static void launchProfiledSolve(Opt_State* state, Opt_Plan* plan, void** problemParams, void** solverParams, std::vector<SolverIteration>& iterationSummary, bool accurateTiming = true) {
    SimpleTimer t;
    t.init();

    Opt_ProblemInit(state, plan, problemParams, solverParams);
    if (accurateTiming) cudaDeviceSynchronize();
    double timeMS = t.tick();
    double cost = Opt_ProblemCurrentCost(state, plan);
    iterationSummary.push_back(SolverIteration(cost, timeMS));

    t.tick();
    while (Opt_ProblemStep(state, plan, problemParams, solverParams)) {
        if (accurateTiming) cudaDeviceSynchronize();
        timeMS = t.tick();
        cost = Opt_ProblemCurrentCost(state, plan);
        iterationSummary.push_back(SolverIteration(cost, timeMS));
        t.tick();
    }
}

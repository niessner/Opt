#ifndef CUDATimer_h
#define CUDATimer_h
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "SolverPerformanceSummary.h"
struct TimingInfo {
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    float duration;
    std::string eventName;
};

#define SYNCHRONIZE_AROUND_EVENTS 0

/** Copied wholesale from mLib, so nvcc doesn't choke. */
template<class T>
int findFirstIndex(const std::vector<T> &collection, const T &value)
{
    int index = 0;
    for (const auto &element : collection)
    {
        if (element == value)
            return index;
        index++;
    }
    return -1;
}

struct RunningStats {
    unsigned int count;
    double sum;
    double sqSum;
    double min;
    double max;
    RunningStats() {}
    RunningStats(double init) :
        count(1), sum(init), sqSum(init*init), min(sum), max(sum) {}
    void update(double newVal) {
        ++count;
        sum += newVal;
        sqSum += (newVal*newVal);
        min = fmin(min, newVal);
        max = fmax(max, newVal);
    }
    SolverPerformanceEntry toPerfEntry() const {
        SolverPerformanceEntry e;
        unsigned int N = count;
        double mean = sum / N;
        double moment2 = sqSum / N;
        double variance = moment2 - (mean*mean);
        double stddev = sqrt(fabs(variance));
        e.count = N;
        e.meanMS = mean;
        e.minMS = min;
        e.maxMS = max;
        e.stddevMS = stddev;
        return e;
    }
};

struct CUDATimer {
    std::vector<TimingInfo> timingEvents;
    std::vector<size_t> activeIndices;
    int currentIteration;

    CUDATimer() : currentIteration(0) {}
    void nextIteration() {
        ++currentIteration;
    }

    void reset() {
        currentIteration = 0;
        timingEvents.clear();
    }

    void startEvent(const std::string& name) {
        TimingInfo timingInfo;
        cudaEventCreate(&timingInfo.startEvent);
        cudaEventCreateWithFlags(&timingInfo.endEvent, cudaEventBlockingSync & 0);
        if (SYNCHRONIZE_AROUND_EVENTS) cudaDeviceSynchronize();
        cudaEventRecord(timingInfo.startEvent);
        timingInfo.eventName = name;
        timingEvents.push_back(timingInfo);
        activeIndices.push_back(timingEvents.size() - 1);
    }

    void endEvent() {
        if (activeIndices.size() == 0) { fprintf(stderr, "ERROR: called endEvent() with no active events!\n"); exit(-1); }
        TimingInfo& timingInfo = timingEvents[activeIndices.back()];
        if (SYNCHRONIZE_AROUND_EVENTS) cudaDeviceSynchronize();
        cudaEventRecord(timingInfo.endEvent, 0);
        activeIndices.pop_back();
    }

    void evaluate(SolverPerformanceSummary& stats) {
        if (activeIndices.size() > 0) {
            printf("WARNING: Evaluating Timing Results, but some events still active!\n");
        }
        while (activeIndices.size() > 0) {
            endEvent();
        }
        std::vector<std::string> aggregateTimingNames;
        std::vector<RunningStats> aggregateStats;
        for (int i = 0; i < timingEvents.size(); ++i) {
            TimingInfo& eventInfo = timingEvents[i];
            cudaEventSynchronize(eventInfo.endEvent);
            cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
            int index = findFirstIndex(aggregateTimingNames, eventInfo.eventName);
            if (index < 0) {
                aggregateTimingNames.push_back(eventInfo.eventName);
                aggregateStats.push_back(RunningStats((double)eventInfo.duration));
            } else {
                aggregateStats[index].update((double)eventInfo.duration);
            }
        }
        printf("|         Kernel          |   Count  | Total(ms) | Average(ms) | Std. Dev(ms) |\n");
        printf("|-------------------------|----------|-----------|-------------|--------------|\n");
        for (int i = 0; i < aggregateTimingNames.size(); ++i) {
            auto e = aggregateStats[i].toPerfEntry();
            printf("| %-23s |   %4u   | %8.3f  |   %8.4f  |   %8.4f   |\n", aggregateTimingNames[i].c_str(),
                e.count, e.meanMS*e.count, e.meanMS, e.stddevMS);
        }
        auto fillOutEntry = [&](std::string name, SolverPerformanceEntry& entry){
            const auto it = std::find(aggregateTimingNames.begin(), aggregateTimingNames.end(), name);
            if (it != aggregateTimingNames.end()) {
                auto index = std::distance(aggregateTimingNames.begin(), it);
                entry = aggregateStats[index].toPerfEntry();
            }
        };
        fillOutEntry("Total", stats.total);
        fillOutEntry("Nonlinear Iteration", stats.nonlinearIteration);
        fillOutEntry("Nonlinear Setup", stats.nonlinearSetup);
        fillOutEntry("Linear Solve", stats.linearSolve);
        fillOutEntry("Nonlinear Finish", stats.nonlinearResolve);

        printf("------------------------------------------------------------\n");
    }
};

#endif
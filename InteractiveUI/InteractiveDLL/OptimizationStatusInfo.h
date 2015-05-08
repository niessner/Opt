#ifndef OptimizationStatusInfo_h
#define OptimizationStatusInfo_h
#include "Optimizer.h"
#include <mutex>
struct OptimizationStatusInfo {
    OptimizationTimingInfo  timingInfo;
    std::string             compilerMessage;
    bool                    currentlyCompiling;
    bool                    informationReadSinceCompilation;
    OptimizationStatusInfo() : currentlyCompiling(false), informationReadSinceCompilation(true) {}
    

};
#endif
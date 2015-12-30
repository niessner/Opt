#ifndef SFSSolverInput_h
#define SFSSolverInput_h
#include "SimpleBuffer.h"
#include "TerraSolverParameters.h"
#include <memory>
#include <string>

struct SFSSolverInput {
    std::shared_ptr<SimpleBuffer>   targetIntensity;
    std::shared_ptr<SimpleBuffer>   targetDepth;
    std::shared_ptr<SimpleBuffer>   previousDepth;
    std::shared_ptr<SimpleBuffer>   initialUnknown; // The values to initialize d_x to before the solver
    std::shared_ptr<SimpleBuffer>   maskEdgeMap; //uint8s, and actually the row and column maps stuck together...
    TerraSolverParameters           parameters;

    void load(const std::string& filenamePrefix) {
        bool onGPU = true;
        targetIntensity = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetIntensity.imagedump", onGPU));
        targetDepth     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetDepth.imagedump",     onGPU));
        previousDepth   = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_previousDepth.imagedump",   onGPU));
        initialUnknown = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_initialUnknown.imagedump", onGPU));
        maskEdgeMap     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_maskEdgeMap.imagedump",     onGPU));
	
        parameters.load(filenamePrefix + ".SFSSolverParameters");
    }

};

#endif

#ifndef SolverInput_h
#define SolverInput_h
#include "SimpleBuffer.h"
#include "TerraSolverParameters.h"
#include <memory>
#include <string>
struct SolverInput {
    shared_ptr<SimpleBuffer> targetIntensity;
    shared_ptr<SimpleBuffer> targetDepth;
    shared_ptr<SimpleBuffer> previousDepth;
    shared_ptr<SimpleBuffer> rowMask;
    shared_ptr<SimpleBuffer> colMask;
    TerraSolverParameters    parameters;

    void load(const std::string& filenamePrefix) {
        bool onGPU = true;
        targetIntensity = shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetIntensity.imagedump", onGPU));
        targetDepth     = shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetDepth.imagedump", onGPU));
        previousDepth   = shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_previousDepth.imagedump", onGPU));
        rowMask         = shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_rowMask.imagedump", onGPU));
        colMask         = shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_colMask.imagedump", onGPU));
        parameters.load(filenamePrefix + ".SFSSolverParameters");
    }

};

#endif
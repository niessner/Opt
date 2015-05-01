#ifndef Optimizer_h
#define Optimizer_h
#include <string>
#include <vector>
#include <OptImage.h>
#include <cuda_runtime.h>
#include <G3D/G3DAll.h>

struct OptimizationInput {
    G3D_DECLARE_ENUM_CLASS(Type, IMAGE, VIDEO, DENSE_GRID, MESH);
    Type type;
    /** JIT produce the target image. */
    shared_ptr<Texture> sourceImage;

    /** Last JIT rendered input to the optimizer based on sourceImage and preprocessing */
    shared_ptr<Texture> lastInput;
    //ImagePreprocessing preprocessing;
    OptimizationInput() : type(Type::IMAGE) {}
    OptimizationInput(shared_ptr<Texture> image) : type(Type::IMAGE), sourceImage(image) {}
    void set(shared_ptr<Texture> image) {
        sourceImage = image;
        type = Type::IMAGE;
    }
};

struct OptimizationOutput {
    G3D_DECLARE_ENUM_CLASS(Type, IMAGE, DENSE_GRID, MESH);
    Type type;
    shared_ptr<Texture> outputImage;
    OptimizationOutput() : type(Type::IMAGE) {}
    /** Initializes an image output */
    void set(int width, int height, int numChannels);
};

struct OptimizationTimingInfo {
    float optDefineTime;
    float optPlanTime;
    float optSolveTime;
    float optSolveTimeGPU;
    OptimizationTimingInfo() : optDefineTime(0), optPlanTime(0), optSolveTime(0), optSolveTimeGPU(0) {}
};
class Optimizer {
private:
    cudaEvent_t m_optSolveStart;
    cudaEvent_t m_optSolveEnd;

    std::vector<OptImage> m_optImages;

public:
    void renderOutput(RenderDevice* rd, const OptimizationOutput& output);
    void setOptData(RenderDevice* rd, Array<OptimizationInput>& input, const OptimizationOutput& output);
    bool run(const std::string& terraFile, const std::string& optimizationMethod, OptimizationTimingInfo& timingInfo, std::string& errorString);


};
#endif
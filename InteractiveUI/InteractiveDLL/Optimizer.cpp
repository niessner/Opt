#include "Optimizer.h"

extern "C" {
#include "Opt.h"
}


void OptimizationOutput::set(int width, int height, int numChannels) {
    outputImage = Texture::createEmpty("Optimization Output", width, height, ImageFormat::floatFormat(numChannels));
    RenderDevice::current->pushState(); {
        RenderDevice::current->setColorClearValue(Color3::black());
        outputImage->clear();
    } RenderDevice::current->popState();
}

void Optimizer::setOptData(RenderDevice* rd, Array<OptimizationInput>& input, const OptimizationOutput& output) {
    m_optImages.clear();
    m_optImages.push_back(OptImage(output.outputImage->width(), output.outputImage->height()));
    
    for (int i = 0; i < input.size(); ++i) {
        OptimizationInput& inputIm = input[i];
        if (isNull(inputIm.lastInput)) {
            inputIm.lastInput = Texture::createEmpty(format("Input %d", i), inputIm.sourceImage->width(), inputIm.sourceImage->height(), ImageFormat::R32F());
        }
        inputIm.lastInput->resize(inputIm.sourceImage->width(), inputIm.sourceImage->height());
        Texture::copy(inputIm.sourceImage, inputIm.lastInput);
        shared_ptr<PixelTransferBuffer> ptb = inputIm.lastInput->toPixelTransferBuffer();
        m_optImages.push_back(OptImage(inputIm.sourceImage->width(), inputIm.sourceImage->height()));
        ptb->getData(const_cast<void*>(m_optImages[m_optImages.size() - 1].DataCPU()));
        // Hack for SFS
        if (inputIm.sourceImage->name() == "Sensor Depth Image") {
            ptb->getData(const_cast<void*>(m_optImages[0].DataCPU()));
        }
    }
    
}

void Optimizer::renderOutput(RenderDevice* rd, const OptimizationOutput& output) {
    shared_ptr<PixelTransferBuffer> ptb = GLPixelTransferBuffer::create(m_optImages[0].dimX, m_optImages[0].dimY, ImageFormat::R32F(), m_optImages[0].dataCPU.data());
    output.outputImage->update(ptb);
}


bool Optimizer::run(const std::string& terraFile, const std::string& optimizationMethod, OptimizationTimingInfo& t, std::string& errorString) {

    t.optDefineTime = 0;
    t.optPlanTime = 0;
    t.optSolveTime = 0;
    t.optSolveTimeGPU = 0;

    if (m_optImages.size() < 2) {
        errorString = "No image available";
        return -1;
    }

    uint64_t dims[] = { m_optImages[0].dimX, m_optImages[0].dimY };

    cudaEventCreate(&m_optSolveStart);
    cudaEventCreate(&m_optSolveEnd);

    Stopwatch timer;
    timer.tick();

    OptState* optimizerState = Opt_NewState();
    if (optimizerState == nullptr)
    {
        errorString = "Opt_NewState failed";
        return -1;
    }

    Problem * prob = Opt_ProblemDefine(optimizerState, terraFile.c_str(), optimizationMethod.c_str(), NULL);
    timer.tock();
    t.optDefineTime = timer.elapsedTime() / units::milliseconds();

    if (!prob)
    {
        errorString = "Opt_ProblemDefine failed";
        return -1;
    }

    timer.tick();

    std::vector<void*> imagesCPU;
    std::vector<void*> imagesGPU;
    std::vector<uint64_t> stride;
    std::vector<uint64_t> elemsize;
    for (const auto &image : m_optImages)
    {
        image.syncCPUToGPU();
        imagesCPU.push_back((void*)image.DataCPU());
        imagesGPU.push_back((void*)image.DataGPU());
        stride.push_back(image.dimX * sizeof(float));
        elemsize.push_back(sizeof(float));
    }

    Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims, elemsize.data(), stride.data());
    timer.tock();
    t.optPlanTime = timer.elapsedTime() / units::milliseconds();

    if (!plan)
    {
        errorString = "Opt_ProblemPlan failed";
        return -1;
    }

    bool isGPU = endsWith(optimizationMethod.c_str(), "GPU");

    timer.tick();
    if (isGPU)
    {
        cudaEventRecord(m_optSolveStart);
        Opt_ProblemSolve(optimizerState, plan, imagesGPU.data(), NULL);
        cudaEventRecord(m_optSolveEnd);
        for (const auto &image : m_optImages)
            image.syncGPUToCPU();
    }
    else {
        Opt_ProblemSolve(optimizerState, plan, imagesCPU.data(), NULL);
    }
    timer.tock();
    t.optSolveTime = timer.elapsedTime() / units::milliseconds();

    cudaEventSynchronize(m_optSolveEnd);
    cudaEventElapsedTime(&t.optSolveTimeGPU, m_optSolveStart, m_optSolveEnd);
    return true;
}
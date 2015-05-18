#include "Optimizer.h"

extern "C" {
#include "Opt.h"
}



void OptimizationInput::set(shared_ptr<ArticulatedModel> am) {
    type = Type::MESH;
    lastInput = shared_ptr<Texture>();
    originalAM = am;

    geoToPositionOffsets.clear();
    for (auto mesh : am->meshArray()) {
        int geoIndex = am->geometryArray().findIndex(mesh->geometry);
        if (!geoToPositionOffsets.containsKey(geoIndex)) {
            geoToPositionOffsets.set(geoIndex, meshGraph.nodes.size());
            for (auto v : mesh->geometry->cpuVertexArray.vertex) {
                meshGraph.nodes.append(Vector4(v.position, 0.0f));
            }
        }  
    }

    meshGraph.connectivity.resize(meshGraph.nodes.size());

    auto addUndirectedEdgePair = [&](int i0, int i1) {
        meshGraph.connectivity[i0].append(i1);
        meshGraph.connectivity[i1].append(i0);
    };

    for (auto mesh : am->meshArray()) {
        int geoIndex = am->geometryArray().findIndex(mesh->geometry);
        int offset;
        geoToPositionOffsets.get(geoIndex, offset);
        for (int i = 0; i < mesh->cpuIndexArray.size(); i += 3) {
            int v0 = offset + mesh->cpuIndexArray[i + 0];
            int v1 = offset + mesh->cpuIndexArray[i + 1];
            int v2 = offset + mesh->cpuIndexArray[i + 2];
            addUndirectedEdgePair(v0, v1);
            addUndirectedEdgePair(v0, v2);
            addUndirectedEdgePair(v1, v2);
        }
    }

    sourceImage = Texture::createEmpty("Model Visualization", 720, 720, ImageFormat::RGBA16F());
}

void OptimizationOutput::set(int width, int height, int numChannels) {
    outputImage = Texture::createEmpty("Optimization Output", width, height, ImageFormat::floatFormat(numChannels));
    //outputImage->visualization.min = 0.35f;
    //outputImage->visualization.min = 0.50f;
    RenderDevice::current->pushState(); {
        RenderDevice::current->setColorClearValue(Color3::black());
        outputImage->clear();
    } RenderDevice::current->popState();
}


/** Creates an optimization output that is duplicate in form from the optimization input */
void OptimizationOutput::set(const OptimizationInput& input) {

    if (input.type == OptimizationInput::Type::IMAGE) {
        type = Type::IMAGE;
        outputImage = Texture::createEmpty("Optimization Output", input.sourceImage->width(), input.sourceImage->height(), ImageFormat::R32F());
        outputImage->visualization.channels = Texture::Visualization::RasL;
    } else if (input.type == OptimizationInput::Type::MESH) {
        type                    = Type::MESH;
        meshGraph               = input.meshGraph;
        geoToPositionOffsets    = input.geoToPositionOffsets;
        // TODO: MAKE A COPY!
        originalAM              = input.originalAM;
        outputImage = Texture::createEmpty("Output Model", input.sourceImage->width(), input.sourceImage->height(), ImageFormat::RGBA16F());
        
        

    } else {
        alwaysAssertM(false, "Given optimization input/output type not yet implemented");
    }

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
    if (output.type == OptimizationOutput::Type::IMAGE) {
        shared_ptr<PixelTransferBuffer> ptb = GLPixelTransferBuffer::create(m_optImages[0].dimX, m_optImages[0].dimY, ImageFormat::R32F(), m_optImages[0].dataCPU.data());
        output.outputImage->update(ptb);
    } else if (output.type == OptimizationOutput::Type::MESH) {
        System::memcpy((void*)output.meshGraph.nodes.getCArray(), m_optImages[0].dataCPU.data(), sizeof(float) * m_optImages[0].dataCPU.size());
        for (int i = 0; i < output.originalAM->geometryArray().size(); ++i) {
            ArticulatedModel::Geometry* geo = output.originalAM->geometryArray()[i];
            int offset;
            output.geoToPositionOffsets.get(i, offset);
            for (int j = 0; j < geo->cpuVertexArray.vertex.size(); ++j) {
                geo->cpuVertexArray.vertex[j].position = output.meshGraph.nodes[j + offset].xyz();
            }
        }
    }
}


bool Optimizer::run(const std::string& terraFile, const std::string& optimizationMethod, OptimizationTimingInfo& t, std::string& errorString) {

    t.optDefineTime = 0;
    t.optPlanTime = 0;
    t.optSolveTime = 0;
    t.optSolveTimeGPU = 0;

    if (m_optImages.size() < 2) {
        errorString = "No image available";
        return false;
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
        return false;
    }

    Problem * prob = Opt_ProblemDefine(optimizerState, terraFile.c_str(), optimizationMethod.c_str(), NULL);
    timer.tock();
    t.optDefineTime = timer.elapsedTime() / units::milliseconds();

    if (!prob)
    {
        errorString = "Opt_ProblemDefine failed";
        return false;
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

    std::vector<int64_t*> adjacencyOffsetsCPU;
    std::vector<int64_t*> adjacencyListsXCPU;
    std::vector<int64_t*> adjacencyListsYCPU;
    std::vector<void*> edgeValuesCPU;

    for (auto &graph : m_optGraphs)
    {
        graph.finalize();
        adjacencyOffsetsCPU.push_back((int64_t *)graph.adjacencyOffsetsCPU.data());
        adjacencyListsXCPU.push_back((int64_t *)graph.adjacencyListsXCPU.data());
        adjacencyListsYCPU.push_back((int64_t *)graph.adjacencyListsYCPU.data());
        edgeValuesCPU.push_back((void*)graph.edgeValuesCPU.data());
    }


    Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims, elemsize.data(), stride.data(), adjacencyOffsetsCPU.data(), adjacencyListsXCPU.data(), adjacencyListsYCPU.data());
    timer.tock();
    t.optPlanTime = timer.elapsedTime() / units::milliseconds();

    if (!plan)
    {
        errorString = "Opt_ProblemPlan failed";
        return false;
    }

    bool isGPU = endsWith(optimizationMethod.c_str(), "GPU");

    timer.tick();
    if (isGPU)
    {
        cudaEventRecord(m_optSolveStart);
        Opt_ProblemSolve(optimizerState, plan, imagesGPU.data(), edgeValuesCPU.data(), NULL);
        cudaEventRecord(m_optSolveEnd);
        for (const auto &image : m_optImages)
            image.syncGPUToCPU();
    }
    else {
        Opt_ProblemSolve(optimizerState, plan, imagesCPU.data(), edgeValuesCPU.data(), NULL);
    }
    timer.tock();
    t.optSolveTime = float(timer.elapsedTime() / units::milliseconds());

    cudaEventSynchronize(m_optSolveEnd);
    cudaEventElapsedTime(&t.optSolveTimeGPU, m_optSolveStart, m_optSolveEnd);
    return true;
}
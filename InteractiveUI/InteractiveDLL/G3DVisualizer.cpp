#include "G3DVisualizer.h"
#include <mLibCore.h>
#include <mLibDepthCamera.h>

#define EMPTY_MODE 0
#define LAPLACIAN_MESH_SMOOTHING 1
#define SHAPE_FROM_SHADING 2
#define DEFAULT_MODE EMPTY_MODE


G3DVisualizer::G3DVisualizer(const GApp::Settings& settings) : GApp(settings), m_initialized(false) {
    renderDevice->setColorClearValue(Color3::white());
}


void G3DVisualizer::moveWindow(int x, int y, int width, int height) {
    window()->setClientRect(Rect2D::xywh(float(x), float(y), float(width), float(height)));
}

void G3DVisualizer::repositionVisualizations() {
    float maxWidth = 400;
    for (int i = 0; i < m_textureBrowserWindows.size() - 1; ++i) {
        float heightToWidthRatio = m_inputs[i].sourceImage->height() / (float)m_inputs[i].sourceImage->width();

        float browserWidth = min(maxWidth, (float)m_inputs[i].sourceImage->width());
        m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        m_textureBrowserWindows[i]->moveTo(Vector2(float(maxWidth + 10) * i, float(window()->height()) / 2 - m_inputs[i].sourceImage->height()));
    }
    if (m_textureBrowserWindows.size() > 0) {
        int i = m_textureBrowserWindows.size() - 1;
        float heightToWidthRatio = m_output.outputImage->height() / (float)m_output.outputImage->width();

        float browserWidth = min(maxWidth, (float)m_output.outputImage->width());
        m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        m_textureBrowserWindows[i]->moveTo(Vector2(0.0f, float(window()->height()) / 2));
    }

}

void G3DVisualizer::visualizeModel(shared_ptr<Texture> outputTexture, shared_ptr<ArticulatedModel> am) {
    RenderDevice* rd = RenderDevice::current;
    static shared_ptr<Framebuffer> fb = Framebuffer::create("Model Visualization Framebuffer");
    static shared_ptr<Texture> depthTexture = Texture::createEmpty("visualizeMode::DEPTH", outputTexture->width(), outputTexture->height(), ImageFormat::DEPTH32());
    static shared_ptr<Framebuffer> depthPeelFrameBuffer = Framebuffer::create(Texture::createEmpty("visualizeMode::DEPTH_PEEL", outputTexture->width(), outputTexture->height(), ImageFormat::DEPTH32()));
    shared_ptr<Texture> depthPeelTex = m_depthPeelFramebuffer->texture(Framebuffer::AttachmentPoint::DEPTH);
    depthPeelTex->resize(outputTexture->width(), outputTexture->height());
    depthTexture->resize(outputTexture->width(), outputTexture->height());
    fb->set(Framebuffer::AttachmentPoint::COLOR0, outputTexture);
    fb->set(Framebuffer::AttachmentPoint::DEPTH, depthTexture);
    Array<shared_ptr<Surface>> surfaces;
    am->pose(surfaces);

    m_gbuffer->setSpecification(m_gbufferSpecification);
    m_gbuffer->resize(fb->width(), fb->height());
    m_gbuffer->prepare(rd, activeCamera(), 0, -(float)previousSimTimeStep(), Vector2int16(0, 0), Vector2int16(0, 0));

    m_renderer->render(rd, fb, depthPeelFrameBuffer, scene()->lightingEnvironment(), m_gbuffer, surfaces);

}

Array<String> indexStrings(int size) {
    Array<String> indexNames;
    for (int i = 0; i < size; ++i) {
        indexNames.append(G3D::format("%d", i));
    }
    if (size < 1) {
        indexNames.append("None");
    }
    return indexNames;
}


void G3DVisualizer::makeGUI() {
    // Initialize the developer HUD (using the existing scene)
    createDeveloperHUD();
    debugWindow->setVisible(true);
    developerWindow->setVisible(false);
    developerWindow->cameraControlWindow->setVisible(false);
    developerWindow->videoRecordDialog->setEnabled(true);

    GuiPane* infoPane = debugPane->addPane("", GuiTheme::ORNATE_PANE_STYLE);
    infoPane->addButton("Load New Input", this, &G3DVisualizer::loadNewInput);
    m_inputDropDownList = infoPane->addDropDownList("Input ", indexStrings(m_inputs.size()), &m_currentInputIndex);
    infoPane->pack();

    debugWindow->pack();
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}

void G3DVisualizer::loadInputFromFile(const String& filename) {

    
    loadInput(filename, m_inputs.size());
    setupVisualizationWidgets();
    
}

void G3DVisualizer::setupVisualizationWidgets() {
    // TODO: MASSIVE CLEANUP
    m_inputDropDownList->setList(indexStrings(m_inputs.size()));
    for (int i = 0; i < m_textureBrowserWindows.size(); ++i) {
        if (notNull(m_textureBrowserWindows[i])) {
            m_textureBrowserWindows[i]->close();
        }
    }
    m_textureBrowserWindows.fastClear();
    for (int i = 0; i < m_inputs.size(); ++i) {
        if (m_inputs[i].type == OptimizationInput::Type::MESH) {
            visualizeModel(m_inputs[i].sourceImage, m_inputs[i].originalAM);
        }

        m_textureBrowserWindows.append(TextureBrowserWindow::create(developerWindow->theme()));
        Array<String> textureNames;
        m_textureBrowserWindows[i]->getTextureList(textureNames);
        m_inputs[i].sourceImage->visualization.documentGamma = 2.2f;
        int texIndex = textureNames.findIndex(m_inputs[i].sourceImage->name());
        m_textureBrowserWindows[i]->setMinSize(Vector2(0, 0));
        m_textureBrowserWindows[i]->setTextureIndex(texIndex);

        float maxWidth = 400;
        float heightToWidthRatio = m_inputs[i].sourceImage->height() / (float)m_inputs[i].sourceImage->width();

        float browserWidth = min(maxWidth, (float)m_inputs[i].sourceImage->width());
        m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        m_textureBrowserWindows[i]->pack();

        addWidget(m_textureBrowserWindows[i]);

        m_textureBrowserWindows[i]->setVisible(true);
        m_textureBrowserWindows[i]->moveTo(Vector2((maxWidth + 10.0f) * i, float(window()->height()) / 2 - m_inputs[i].sourceImage->height()));
    }
    int i = m_textureBrowserWindows.size();
    m_textureBrowserWindows.append(TextureBrowserWindow::create(developerWindow->theme()));
    Array<String> textureNames;

    m_textureBrowserWindows[i]->getTextureList(textureNames);
    m_output.outputImage->visualization.documentGamma = 2.2f;
    if (m_output.type == OptimizationOutput::Type::IMAGE) {
        m_output.outputImage->visualization.channels = Texture::Visualization::RasL;
    }
    int texIndex = textureNames.findIndex(m_output.outputImage->name());
    m_textureBrowserWindows[i]->setMinSize(Vector2(0, 0));
    m_textureBrowserWindows[i]->setTextureIndex(texIndex);

    float maxWidth = 400;
    float heightToWidthRatio = m_output.outputImage->height() / (float)m_output.outputImage->width();

    float browserWidth = min(maxWidth, (float)m_output.outputImage->width());
    m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));
    m_textureBrowserWindows[i]->pack();

    addWidget(m_textureBrowserWindows[i]);
    m_textureBrowserWindows[i]->setVisible(true);

    m_repositionVisualizationNextFrame = true;
}

void G3DVisualizer::loadNewInput() {
    if (FileDialog::getFilename(m_currentFilename, "png,sensor,Any", false)) {
        loadInputFromFile(m_currentFilename);
    }
}

void G3DVisualizer::onInit() {
    GApp::onInit();
    m_currentInputIndex = 0;
    m_initialized = true;
    m_outputWidth = 1;
    m_outputHeight = 1;
    m_outputChannelCount = 1;

    setLowerFrameRateInBackground(false);
    makeGUI();
    m_repositionVisualizationNextFrame = false;
    showRenderingStats = false;


    loadScene(System::findDataFile("default.Scene.Any"));
    setActiveCamera(m_debugCamera);
#if DEFAULT_MODE == SHAPE_FROM_SHADING
    loadInputFromFile("E:/Projects/DSL/Optimization/API/testMLib/refined.sensor");
    //loadInputFromFile("E:/Projects/DSL/Optimization/API/testMLib/recordingRaw.sensor");
    generateSFSInput(m_inputs[1].sourceImage, m_inputs[0].sourceImage);
#elif DEFAULT_MODE == LAPLACIAN_MESH_SMOOTHING
    loadInputFromFile(System::findDataFile("model/buddha/buddha.ArticulatedModel.Any"));
#endif
    

}

void G3DVisualizer::loadImage(const String& filename, int inputIndex) {
    alwaysAssertM(inputIndex <= m_inputs.size(), "Tried to add input to invalid index");
    if (inputIndex == m_inputs.size()) {
        m_inputs.resize(inputIndex + 1);
    }
    // Assumes image for now
    shared_ptr<Texture> inputTexture = Texture::fromFile(filename);
    m_inputs[inputIndex].set(inputTexture);
    if (m_inputs.size() == 1) {
        m_output.set(inputTexture->width(), inputTexture->height(), 1);
    }
}

void G3DVisualizer::loadArticulatedModel(const String& filename, int inputIndex) {
    alwaysAssertM(inputIndex <= m_inputs.size(), "Tried to add input to invalid index");
    if (inputIndex == m_inputs.size()) {
        m_inputs.resize(inputIndex + 1);
    }
    Any spec;
    spec.load(filename);
    shared_ptr<ArticulatedModel> am = ArticulatedModel::create(spec, "Input Model");
    m_inputs[inputIndex].set(am);
    if (m_inputs.size() == 1) {
        m_output.set(m_inputs[0]);
        m_output.originalAM = ArticulatedModel::create(spec, "Output Model");
    }
}


void G3DVisualizer::loadDepthColorFrame(const String& filename, int inputIndex) {
    alwaysAssertM(inputIndex <= m_inputs.size() + 1, "Tried to add input to invalid index");
    if (inputIndex >= m_inputs.size()) {
        m_inputs.resize(inputIndex + 2);
    }


    ml::BinaryDataStreamFile inputStream(filename.c_str(), false);
    ml::CalibratedSensorData sensorData;
    inputStream >> sensorData;
    for (unsigned int i = 0; i < sensorData.m_DepthImageHeight*sensorData.m_DepthImageWidth; ++i) {
        //sensorData.m_DepthImages[0][i] += Random::common().uniform()*0.2f;
    }
    shared_ptr<Texture> depthTexture = Texture::fromPixelTransferBuffer("Initial Sensor Depth Image",
        CPUPixelTransferBuffer::fromData(sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, ImageFormat::R32F(), sensorData.m_DepthImages[0]));
    

    shared_ptr<Texture> colorTexture = Texture::fromPixelTransferBuffer("Initial Sensor Color Image",
        CPUPixelTransferBuffer::fromData(sensorData.m_ColorImageWidth, sensorData.m_ColorImageHeight, ImageFormat::BGRA8(), sensorData.m_ColorImages[0]));
    
    shared_ptr<Texture> resampledDepth = Texture::createEmpty("Sensor Depth Image", sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, ImageFormat::R32F());
    shared_ptr<Texture> resampledColor = Texture::createEmpty("Sensor Color Image", sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, ImageFormat::RGBA32F());
    m_sfs.resampleImages(depthTexture, colorTexture, resampledDepth, resampledColor);
    resampledDepth->visualization.channels = Texture::Visualization::RasL;
    resampledDepth->visualization.min = 0.35f;
    resampledDepth->visualization.max = 0.50f;
    resampledColor->visualization.documentGamma = 2.2f;

    m_inputs[inputIndex].set(resampledDepth);
    m_inputs[inputIndex + 1].set(resampledColor);
    if (m_inputs.size() == 2) {
        m_output.set(resampledDepth->width(), resampledDepth->height(), 1);
    }
}

void G3DVisualizer::generateSFSInput(shared_ptr<Texture> color, shared_ptr<Texture> depth) {
    Array<float> lightingCoefficients;
    static shared_ptr<Texture> outputAlbedo = Texture::createEmpty("Albedo Texture", color->width(), color->height(), ImageFormat::RGBA32F());
    static shared_ptr<Texture> targetLuminance = Texture::createEmpty("Target Luminance", color->width(), color->height(), ImageFormat::R32F());
    shared_ptr<Texture> albedoLuminance = Texture::createEmpty("Albedo Luminance Texture", color->width(), color->height(), ImageFormat::R32F());
    m_sfs.estimateLightingAndAlbedo(color, depth, outputAlbedo, targetLuminance, albedoLuminance, lightingCoefficients);
    for (int i = 0; i < lightingCoefficients.size(); ++i) {
        debugPrintf("%d: %f\n", i, lightingCoefficients[i]);
    }
    targetLuminance->visualization.channels = Texture::Visualization::RasL;
    targetLuminance->visualization.documentGamma = 2.2f;
    albedoLuminance->visualization.channels = Texture::Visualization::RasL;
    albedoLuminance->visualization.documentGamma = 2.2f;

    m_inputs[m_inputs.size() - 1].set(targetLuminance);
    m_inputs.resize(m_inputs.size() + 1);
    m_inputs[m_inputs.size() - 1].set(albedoLuminance);
    m_output.outputImage->visualization = Texture::Visualization::RasL;
    m_output.outputImage->visualization.documentGamma = 2.2f;
    m_output.outputImage->visualization.max = 2.0f;
    setupVisualizationWidgets();
}

void G3DVisualizer::loadInput(const String& filename, int inputIndex) {
    if (endsWith(filename, ".sensor")) {
        loadDepthColorFrame(filename, inputIndex);
    } else if (endsWith(filename, ".ArticulatedModel.Any")) {
        loadArticulatedModel(filename, inputIndex);
    } else {
        loadImage(filename, inputIndex);
    }
}


bool G3DVisualizer::onEvent(const GEvent& e) {
    if (GApp::onEvent(e)) {
        return true;
    }
    // If you need to track individual UI events, manage them here.
    // Return true if you want to prevent other parts of the system
    // from observing this specific event.
    //
    // For example,
    // if ((e.type == GEventType::GUI_ACTION) && (e.gui.control == m_button)) { ... return true;}
    // if ((e.type == GEventType::KEY_DOWN) && (e.key.keysym.sym == GKey::TAB)) { ... return true; }

    return false;
}

void G3DVisualizer::cudaSFS() {

}


void G3DVisualizer::onGraphics3D(RenderDevice* rd, Array<shared_ptr<Surface> >& surface3D) {
    rd->swapBuffers();
    for (int i = 0; i < m_inputs.size(); ++i) {
        if (m_inputs[i].type == OptimizationInput::Type::MESH) {
            visualizeModel(m_inputs[i].sourceImage, m_inputs[i].originalAM);
        }
    }
    if (m_output.type == OptimizationOutput::Type::MESH) {
        visualizeModel(m_output.outputImage, m_output.originalAM);
    }



    rd->clear();


    alwaysAssertM(m_outputVisualizationMode == VisualizationMode::IMAGE, "Only image visualization is currently supported.");
    static shared_ptr<GFont> font = GFont::fromFile(System::findDataFile("arial.fnt"));
    rd->push2D(); {
        if (m_inputs.size() == 0) {
            font->draw2D(rd, "No inputs", Point2(window()->width() / 2.0f, window()->height() / 2.0f), 60.0f, Color3::black(), Color4::clear(), GFont::XAlign::XALIGN_CENTER, GFont::YAlign::YALIGN_CENTER);
        } else {
            /*
            Draw::rect2D(Rect2D::xywh(0, window()->height() / 2 - 1, window()->width(), 2), rd, Color3::black());
            
            shared_ptr<Texture> im = m_inputs[m_currentInputIndex].sourceImage;
            Draw::rect2D(Rect2D::xywh(0, window()->height() / 2 - 1 - im->height(), im->width(), im->height()), rd, Color3::white(), im); 
            
            im = m_output.outputImage;
            Draw::rect2D(Rect2D::xywh(0, window()->height() / 2 + 1, im->width(), im->height()), rd, Color3::white(), im);
            */
        }
    } rd->pop2D();
}


void G3DVisualizer::onGraphics2D(RenderDevice* rd, Array<Surface2D::Ref>& posed2D) {
    // Render 2D objects like Widgets.  These do not receive tone mapping or gamma correction
    Surface2D::sortAndRender(rd, posed2D);
}

void G3DVisualizer::onSimulation(RealTime rdt, SimTime sdt, SimTime idt) {
    GApp::onSimulation(rdt, sdt, idt);

    executeMoveMessage();

    if (m_repositionVisualizationNextFrame) {
        repositionVisualizations();
        m_repositionVisualizationNextFrame = false;
    }

    RealTime lastMessageTime;
    std::string terraFile, optimizationMethod, errorString;
    getRunOptMessage(terraFile, optimizationMethod, lastMessageTime);
    if (lastMessageTime > m_lastOptimizationRunTime) {
        
        m_lastOptimizationRunTime = lastMessageTime;
        OptimizationTimingInfo timingInfo;
        m_optimizer.setOptData(RenderDevice::current, m_inputs, m_output);
        signalCompilation();
        m_optimizer.run(terraFile, optimizationMethod, timingInfo, errorString);
        writeStatusInfo(timingInfo, errorString);
        m_optimizer.renderOutput(RenderDevice::current, m_output);
        static shared_ptr<Texture> diffTexture = Texture::createEmpty("Diff Tex", m_output.outputImage->width(), m_output.outputImage->height(), ImageFormat::RG32F());
        diffTexture->visualization.channels = Texture::Visualization::RasL;
        static shared_ptr<Framebuffer> fb = Framebuffer::create(diffTexture);
        RenderDevice::current->push2D(fb); {
            Args args;
            m_inputs[0].sourceImage->setShaderArgs(args, "input0_", Sampler::buffer());
            m_output.outputImage->setShaderArgs(args, "input1_", Sampler::buffer());
            args.setRect(RenderDevice::current->viewport());
            LAUNCH_SHADER("diff.pix", args);
        } RenderDevice::current->pop2D();
       

    }

    // Example GUI dynamic layout code.  Resize the debugWindow to fill
    // the screen horizontally.
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}
#include "G3DVisualizer.h"
#include <mLibCore.h>
#include <mLibDepthCamera.h>

G3DVisualizer::G3DVisualizer(const GApp::Settings& settings) : GApp(settings), m_initialized(false) {
    renderDevice->setColorClearValue(Color3::white());
}


void G3DVisualizer::moveWindow(int x, int y, int width, int height) {
    window()->setClientRect(Rect2D::xywh(x, y, width, height));
}

void G3DVisualizer::repositionVisualizations() {
    float maxWidth = 400;
    for (int i = 0; i < m_textureBrowserWindows.size() - 1; ++i) {
        float heightToWidthRatio = m_inputs[i].sourceImage->height() / (float)m_inputs[i].sourceImage->width();

        float browserWidth = min(maxWidth, (float)m_inputs[i].sourceImage->width());
        m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        m_textureBrowserWindows[i]->moveTo(Vector2((maxWidth + 10) * i, window()->height() / 2 - m_inputs[i].sourceImage->height()));
    }
    if (m_textureBrowserWindows.size() > 0) {
        int i = m_textureBrowserWindows.size() - 1;
        float heightToWidthRatio = m_output.outputImage->height() / (float)m_output.outputImage->width();

        float browserWidth = min(maxWidth, (float)m_output.outputImage->width());
        m_textureBrowserWindows[i]->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        m_textureBrowserWindows[i]->moveTo(Vector2(0, window()->height() / 2));
    }
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

    // TODO: MASSIVE CLEANUP
    loadInput(filename, m_inputs.size());
    m_inputDropDownList->setList(indexStrings(m_inputs.size()));
    for (int i = 0; i < m_textureBrowserWindows.size(); ++i) {
        if (notNull(m_textureBrowserWindows[i])) {
            m_textureBrowserWindows[i]->close();
        }
    }
    m_textureBrowserWindows.fastClear();
    for (int i = 0; i < m_inputs.size(); ++i) {
        m_textureBrowserWindows.append(TextureBrowserWindow::create(developerWindow->theme()));
        Array<String> textureNames;
        m_textureBrowserWindows[i]->getTextureList(textureNames);
        m_inputs[i].sourceImage->visualization.documentGamma = 2.2;
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
        m_textureBrowserWindows[i]->moveTo(Vector2((maxWidth + 10) * i, window()->height() / 2 - m_inputs[i].sourceImage->height()));
    }
    int i = m_textureBrowserWindows.size();
    m_textureBrowserWindows.append(TextureBrowserWindow::create(developerWindow->theme()));
    Array<String> textureNames;

    m_textureBrowserWindows[i]->getTextureList(textureNames);
    m_output.outputImage->visualization.documentGamma = 2.2;
    m_output.outputImage->visualization.channels = Texture::Visualization::RasL;
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
    if (FileDialog::getFilename(m_currentFilename, "png,sensor", false)) {
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
    loadInputFromFile("E:/Projects/DSL/Optimization/API/testMLib/recordingRaw.sensor");

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

void G3DVisualizer::loadDepthColorFrame(const String& filename, int inputIndex) {
    alwaysAssertM(inputIndex <= m_inputs.size() + 1, "Tried to add input to invalid index");
    if (inputIndex >= m_inputs.size()) {
        m_inputs.resize(inputIndex + 2);
    }


    ml::BinaryDataStreamFile inputStream(filename.c_str(), false);
    ml::CalibratedSensorData sensorData;
    inputStream >> sensorData;
    for (int i = 0; i < sensorData.m_DepthImageHeight*sensorData.m_DepthImageWidth; ++i) {
        sensorData.m_DepthImages[0][i] += Random::common().uniform()*0.2;
    }
    shared_ptr<Texture> depthTexture = Texture::fromPixelTransferBuffer("Sensor Depth Image",
        CPUPixelTransferBuffer::fromData(sensorData.m_DepthImageWidth, sensorData.m_DepthImageHeight, ImageFormat::R32F(), sensorData.m_DepthImages[0]));
    depthTexture->visualization.channels = Texture::Visualization::RasL;

    shared_ptr<Texture> colorTexture = Texture::fromPixelTransferBuffer("Sensor Color Image",
        CPUPixelTransferBuffer::fromData(sensorData.m_ColorImageWidth, sensorData.m_ColorImageHeight, ImageFormat::BGRA8(), sensorData.m_ColorImages[0]));
    colorTexture->visualization.documentGamma = 2.2f;

    m_inputs[inputIndex].set(depthTexture);
    m_inputs[inputIndex+1].set(colorTexture);
    if (m_inputs.size() == 2) {
        m_output.set(depthTexture->width(), depthTexture->height(), 1);
    }
}

void G3DVisualizer::loadInput(const String& filename, int inputIndex) {
    if (endsWith(filename, ".sensor")) {
        loadDepthColorFrame(filename, inputIndex);
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


void G3DVisualizer::onGraphics3D(RenderDevice* rd, Array<shared_ptr<Surface> >& surface3D) {
    rd->swapBuffers();
    rd->clear();


    alwaysAssertM(m_outputVisualizationMode == VisualizationMode::IMAGE, "Only image visualization is currently supported.");
    static shared_ptr<GFont> font = GFont::fromFile(System::findDataFile("arial.fnt"));
    rd->push2D(); {
        if (m_inputs.size() == 0) {
            font->draw2D(rd, "No inputs", Point2(window()->width() / 2, window()->height() / 2), 60.0, Color3::black(), Color4::clear(), GFont::XAlign::XALIGN_CENTER, GFont::YAlign::YALIGN_CENTER);
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
    }

    // Example GUI dynamic layout code.  Resize the debugWindow to fill
    // the screen horizontally.
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}
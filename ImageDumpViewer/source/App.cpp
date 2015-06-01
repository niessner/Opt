/** \file App.cpp */
#include "App.h"

// Tells C++ to invoke command-line main() function even on OS X and Win32.
G3D_START_AT_MAIN();

int main(int argc, const char* argv[]) {
    {
        G3DSpecification g3dSpec;
        g3dSpec.audio = false;
        initGLG3D(g3dSpec);
    }

    GApp::Settings settings(argc, argv);

    // Change the window and other startup parameters by modifying the
    // settings class.  For example:
    settings.window.caption             = argv[0];
    // settings.window.debugContext     = true;

    // settings.window.width              =  854; settings.window.height       = 480;
    // settings.window.width            = 1024; settings.window.height       = 768;
     settings.window.width            = 1280; settings.window.height       = 720;
//    settings.window.width               = 1920; settings.window.height       = 1080;
    // settings.window.width            = OSWindow::primaryDisplayWindowSize().x; settings.window.height = OSWindow::primaryDisplayWindowSize().y;
    settings.window.fullScreen          = false;
    settings.window.resizable           = ! settings.window.fullScreen;
    settings.window.framed              = ! settings.window.fullScreen;

    // Set to true for a significant performance boost if your app can't render at 60fps,
    // or if you *want* to render faster than the display.
    settings.window.asynchronous        = false;

    settings.depthGuardBandThickness    = Vector2int16(64, 64);
    settings.colorGuardBandThickness    = Vector2int16(16, 16);
    settings.dataDir                    = FileSystem::currentDirectory();
    settings.screenshotDirectory        = "../journal/";

    return App(settings).run();
}


App::App(const GApp::Settings& settings) : GApp(settings) {
}


void App::loadImageDump(const String& filename) {
    BinaryInput bi(filename, G3D_LITTLE_ENDIAN);
    int width   = bi.readInt32();
    int height = bi.readInt32();
    int channels = bi.readInt32();
    int datatype = bi.readInt32();
    Array<float> dataArray;
    dataArray.resize(width*height*channels);
    for (int i = 0; i < width*height*channels; ++i) {
        dataArray[i] = bi.readFloat32();
    }
    //bi.readFloat32(dataArray, width*height*channels);
    alwaysAssertM(datatype == 0, "Only float handling currently implemented");
    shared_ptr<GLPixelTransferBuffer> ptb = GLPixelTransferBuffer::create(width, height, ImageFormat::floatFormat(channels), dataArray.getCArray());
    shared_ptr<Texture> newTexture = Texture::createEmpty(filename, width, height, ImageFormat::floatFormat(channels));
    newTexture->update(ptb);
    newTexture->visualization.channels = Texture::Visualization::RasL;
    newTexture->visualization.max = 2;
    newTexture->visualization.min = -1;
    m_dumpTextures.append(newTexture);


    /*if (G3D::endsWith(newTexture->name(), "JTF_cuda.imagedump") || G3D::endsWith(newTexture->name(), "JTJ_cuda.imagedump") || G3D::endsWith(newTexture->name(), "Pre_cuda.imagedump")){
        static shared_ptr<TextureBrowserWindow> w = TextureBrowserWindow::create(developerWindow->theme());
        Array<String> textureNames;
        w->getTextureList(textureNames);
        int texIndex = textureNames.findIndex(newTexture->name());
        w->setMinSize(Vector2(0, 0));
        w->setTextureIndex(texIndex);

        float maxWidth = 400;
        float heightToWidthRatio = newTexture->height() / (float)newTexture->width();

        float browserWidth = min(maxWidth, (float)newTexture->width());
        w->textureBox()->setSizeFromInterior(Vector2(browserWidth, browserWidth * heightToWidthRatio));

        w->pack();

        addWidget(w);

        w->setVisible(true);
    }*/

}

static shared_ptr<Texture> quotientImage(RenderDevice* rd, const String& name, shared_ptr<Texture> t0, shared_ptr<Texture> t1) {
    shared_ptr<Framebuffer> fb = Framebuffer::create(Texture::createEmpty(name, t0->width(), t1->height(), ImageFormat::RGBA32F()));
    fb->texture(0)->visualization.documentGamma = 2.2f;
    rd->push2D(fb); {
        Args args;
        args.setUniform("input0_buffer", t0, Sampler::buffer());
        args.setUniform("input1_buffer", t1, Sampler::buffer());
        args.setRect(rd->viewport());
        LAUNCH_SHADER("imageQuotient.pix", args);
    } rd->pop2D();
    return fb->texture(0);
}

static shared_ptr<Texture> differenceImage(RenderDevice* rd, const String& name, shared_ptr<Texture> t0, shared_ptr<Texture> t1) {
    shared_ptr<Framebuffer> fb = Framebuffer::create(Texture::createEmpty(name, t0->width(), t1->height(), ImageFormat::RGBA32F()));
    fb->texture(0)->visualization.documentGamma = 2.2f;
    rd->push2D(fb); {
        Args args;
        args.setUniform("input0_buffer", t0, Sampler::buffer());
        args.setUniform("input1_buffer", t1, Sampler::buffer());
        args.setRect(rd->viewport());
        LAUNCH_SHADER("imageDifference.pix", args);
    } rd->pop2D();
    return fb->texture(0);
}



// Called before the application loop begins.  Load data here and
// not in the constructor so that common exceptions will be
// automatically caught.
void App::onInit() {
    GApp::onInit();
    setFrameDuration(1.0f / 120.0f);

    // Call setScene(shared_ptr<Scene>()) or setScene(MyScene::create()) to replace
    // the default scene here.
    
    showRenderingStats      = true;

    makeGUI();
    // For higher-quality screenshots:
    // developerWindow->videoRecordDialog->setScreenShotFormat("PNG");
    // developerWindow->videoRecordDialog->setCaptureGui(false);
    developerWindow->cameraControlWindow->moveTo(Point2(developerWindow->cameraControlWindow->rect().x0(), 0));
    
    String directory = "E:/Projects/DSL/Optimization/Examples/ImageWarping/";

    Array<String> filenames;
    FileSystem::getFiles(directory+"*.imagedump", filenames);
    for (auto f : filenames) {
        loadImageDump(directory + f);
    }
    
    RenderDevice* rd = RenderDevice::current;
    
    shared_ptr<Texture> jtfCuda = Texture::getTextureByName(directory + "JTF_AD.imagedump");
    shared_ptr<Texture> jtfOptNoAD = Texture::getTextureByName(directory + "JTF_optNoAD.imagedump");
    

    shared_ptr<Texture> costCuda = Texture::getTextureByName(directory + "cost_AD.imagedump");
    shared_ptr<Texture> costOptNoAD = Texture::getTextureByName(directory + "cost_optNoAD.imagedump");
    

    static shared_ptr<Texture> preDiff = Texture::singleChannelDifference(rd,
        Texture::getTextureByName(directory + "Pre_AD.imagedump"),
        Texture::getTextureByName(directory + "Pre_optNoAD.imagedump"));

    shared_ptr<Texture> jtjCuda = Texture::getTextureByName(directory + "JTJ_AD.imagedump");
    shared_ptr<Texture> jtjOptNoAD = Texture::getTextureByName(directory + "JTJ_optNoAD.imagedump");

    

    shared_ptr<Texture> resultOpt = Texture::getTextureByName(directory + "result_AD.imagedump");
    shared_ptr<Texture> resultOptNoAD = Texture::getTextureByName(directory + "result_optNoAD.imagedump");

    static shared_ptr<Texture> jtfDiff = differenceImage(rd, "JTF Difference", jtfCuda, jtfOptNoAD);
    static shared_ptr<Texture> costDiff = Texture::singleChannelDifference(rd, costCuda, costOptNoAD);
    static shared_ptr<Texture> jtjDiff = differenceImage(rd, "JTJ Difference", jtjCuda, jtjOptNoAD);
    static shared_ptr<Texture> resultDiff = differenceImage(rd, "Result Difference", resultOpt, resultOptNoAD);
    jtfDiff->visualization.documentGamma = 2.2f;
    costDiff->visualization.documentGamma = 2.2f;
    jtjDiff->visualization.documentGamma = 2.2f;
    resultDiff->visualization.documentGamma = 2.2f;

    /*
    static shared_ptr<Texture> initDiff = Texture::singleChannelDifference(rd, Texture::getTextureByName("E:/Projects/DSL/Optimization/API/RealtimeSFS/initial_depth.imagedump"), resultOpt);
    static shared_ptr<Texture> initDiff2 = Texture::singleChannelDifference(rd, Texture::getTextureByName("E:/Projects/DSL/Optimization/API/RealtimeSFS/initial_depth.imagedump"), resultOptNoAD);
    initDiff->visualization.documentGamma = 2.2f;
    initDiff2->visualization.documentGamma = 2.2f;
    */

    static shared_ptr<Texture> jtjQ = quotientImage(rd, "JTJ Quotient", jtjCuda, jtjOptNoAD);
    static shared_ptr<Texture> jtfQ = quotientImage(rd, "JTF Quotient", jtfCuda, jtfOptNoAD);


    dynamic_pointer_cast<DefaultRenderer>(m_renderer)->setOrderIndependentTransparency(false);
}


void App::makeGUI() {
    // Initialize the developer HUD (using the existing scene)
    createDeveloperHUD();
    debugWindow->setVisible(true);
    developerWindow->videoRecordDialog->setEnabled(true);

    GuiPane* infoPane = debugPane->addPane("Info", GuiTheme::ORNATE_PANE_STYLE);

    // Example of how to add debugging controls
    infoPane->addLabel("You can add GUI controls");
    infoPane->addLabel("in App::onInit().");
    infoPane->addButton("Exit", this, &App::endProgram);
    infoPane->pack();

    // More examples of debugging GUI controls:
    // debugPane->addCheckBox("Use explicit checking", &explicitCheck);
    // debugPane->addTextBox("Name", &myName);
    // debugPane->addNumberBox("height", &height, "m", GuiTheme::LINEAR_SLIDER, 1.0f, 2.5f);
    // button = debugPane->addButton("Run Simulator");

    debugWindow->pack();
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}


void App::onGraphics3D(RenderDevice* rd, Array<shared_ptr<Surface> >& allSurfaces) {
    // This implementation is equivalent to the default GApp's. It is repeated here to make it
    // easy to modify rendering. If you don't require custom rendering, just delete this
    // method from your application and rely on the base class.

    if (! scene()) {
        return;
    }

    m_gbuffer->setSpecification(m_gbufferSpecification);
    m_gbuffer->resize(m_framebuffer->width(), m_framebuffer->height());
    m_gbuffer->prepare(rd, activeCamera(), 0, -(float)previousSimTimeStep(), m_settings.depthGuardBandThickness, m_settings.colorGuardBandThickness);

    m_renderer->render(rd, m_framebuffer, m_depthPeelFramebuffer, scene()->lightingEnvironment(), m_gbuffer, allSurfaces);

    // Debug visualizations and post-process effects
    rd->pushState(m_framebuffer); {
        // Call to make the App show the output of debugDraw(...)
        drawDebugShapes();
        const shared_ptr<Entity>& selectedEntity = (notNull(developerWindow) && notNull(developerWindow->sceneEditorWindow)) ? developerWindow->sceneEditorWindow->selectedEntity() : shared_ptr<Entity>();
        //scene()->visualize(rd, selectedEntity, sceneVisualizationSettings());

        // Post-process special effects
        m_depthOfField->apply(rd, m_framebuffer->texture(0), m_framebuffer->texture(Framebuffer::DEPTH), activeCamera(), m_settings.depthGuardBandThickness - m_settings.colorGuardBandThickness);
        
        m_motionBlur->apply(rd, m_framebuffer->texture(0), m_gbuffer->texture(GBuffer::Field::SS_EXPRESSIVE_MOTION), 
                            m_framebuffer->texture(Framebuffer::DEPTH), activeCamera(), 
                            m_settings.depthGuardBandThickness - m_settings.colorGuardBandThickness);
    } rd->popState();

    if ((bufferSwapMode() == BufferSwapMode::EXPLICIT) && (!renderDevice->swapBuffersAutomatically())) {
        // We're about to render to the actual back buffer, so swap the buffers now.
        // This call also allows the screenshot and video recording to capture the
        // previous frame just before it is displayed.
        swapBuffers();
    }

	// Clear the entire screen (needed even though we'll render over it, since
    // AFR uses clear() to detect that the buffer is not re-used.)
    rd->clear();

    // Perform gamma correction, bloom, and SSAA, and write to the native window frame buffer
    m_film->exposeAndRender(rd, activeCamera()->filmSettings(), m_framebuffer->texture(0));
}


void App::onAI() {
    GApp::onAI();
    // Add non-simulation game logic and AI code here
}


void App::onNetwork() {
    GApp::onNetwork();
    // Poll net messages here
}


void App::onSimulation(RealTime rdt, SimTime sdt, SimTime idt) {
    GApp::onSimulation(rdt, sdt, idt);

    // Example GUI dynamic layout code.  Resize the debugWindow to fill
    // the screen horizontally.
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}


bool App::onEvent(const GEvent& event) {
    // Handle super-class events
    if (GApp::onEvent(event)) { return true; }

    // If you need to track individual UI events, manage them here.
    // Return true if you want to prevent other parts of the system
    // from observing this specific event.
    //
    // For example,
    // if ((event.type == GEventType::GUI_ACTION) && (event.gui.control == m_button)) { ... return true; }
    // if ((event.type == GEventType::KEY_DOWN) && (event.key.keysym.sym == GKey::TAB)) { ... return true; }

    return false;
}


void App::onUserInput(UserInput* ui) {
    GApp::onUserInput(ui);
    (void)ui;
    // Add key handling here based on the keys currently held or
    // ones that changed in the last frame.
}


void App::onPose(Array<shared_ptr<Surface> >& surface, Array<shared_ptr<Surface2D> >& surface2D) {
    GApp::onPose(surface, surface2D);

    // Append any models to the arrays that you want to later be rendered by onGraphics()
}


void App::onGraphics2D(RenderDevice* rd, Array<shared_ptr<Surface2D> >& posed2D) {
    // Render 2D objects like Widgets.  These do not receive tone mapping or gamma correction.
    Surface2D::sortAndRender(rd, posed2D);
}


void App::onCleanup() {
    // Called after the application loop ends.  Place a majority of cleanup code
    // here instead of in the constructor so that exceptions can be caught.
}


void App::endProgram() {
    m_endProgram = true;
}

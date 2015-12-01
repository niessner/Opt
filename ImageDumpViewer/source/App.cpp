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



static void compareCUDABlockAndNonBlock(RenderDevice* rd, const String& directory) {
    String nonBlock = "_nonblock_cuda.imagedump";
    String block = "_block_cuda.imagedump";

    shared_ptr<Texture> jtfBlock = Texture::getTextureByName(directory + "JTF" + block);
    shared_ptr<Texture> jtfNBlock = Texture::getTextureByName(directory + "JTF" + nonBlock);


    shared_ptr<Texture> costBlock = Texture::getTextureByName(directory + "cost" + block);
    shared_ptr<Texture> costNBlock = Texture::getTextureByName(directory + "cost" + nonBlock);


    static shared_ptr<Texture> preDiff = Texture::singleChannelDifference(rd,
        Texture::getTextureByName(directory + "Pre" + block),
        Texture::getTextureByName(directory + "Pre" + nonBlock));

    shared_ptr<Texture> jtjBlock = Texture::getTextureByName(directory + "JTJ" + block);
    shared_ptr<Texture> jtjNBlock = Texture::getTextureByName(directory + "JTJ" + nonBlock);
    static shared_ptr<Texture> jtfDiff = differenceImage(rd, "JTF Difference", jtfBlock, jtfNBlock);
    static shared_ptr<Texture> costDiff = Texture::singleChannelDifference(rd, costBlock, costNBlock);
    static shared_ptr<Texture> jtjDiff = differenceImage(rd, "JTJ Difference", jtjBlock, jtjNBlock);

    jtfDiff->visualization.documentGamma = 2.2f;
    costDiff->visualization.documentGamma = 2.2f;
    jtjDiff->visualization.documentGamma = 2.2f;

    static shared_ptr<Texture> jtjQ = quotientImage(rd, "JTJ Quotient", jtjBlock, jtjNBlock);
    static shared_ptr<Texture> jtfQ = quotientImage(rd, "JTF Quotient", jtfBlock, jtfNBlock);
    static shared_ptr<Texture> costQ = quotientImage(rd, "Cost Quotient", costBlock, costNBlock);
}

static void compareCUDAAndTerraNonBlock(RenderDevice* rd, const String& directory) {
    String cuda = "_nonblock_cuda.imagedump";
    String terra = "_optNoAD.imagedump";

    shared_ptr<Texture> jtfCUDA = Texture::getTextureByName(directory + "JTF" + cuda);
    shared_ptr<Texture> jtfTerra = Texture::getTextureByName(directory + "JTF" + terra);


    shared_ptr<Texture> costCUDA = Texture::getTextureByName(directory + "cost" + cuda);
    shared_ptr<Texture> costTerra = Texture::getTextureByName(directory + "cost" + terra);


    static shared_ptr<Texture> preDiff = Texture::singleChannelDifference(rd,
        Texture::getTextureByName(directory + "Pre" + cuda),
        Texture::getTextureByName(directory + "Pre" + terra));

    shared_ptr<Texture> jtjCUDA = Texture::getTextureByName(directory + "JTJ" + cuda);
    shared_ptr<Texture> jtjTerra = Texture::getTextureByName(directory + "JTJ" + terra);

    static shared_ptr<Texture> jtfDiff = differenceImage(rd, "JTF Difference", jtfCUDA, jtfTerra);
    static shared_ptr<Texture> costDiff = Texture::singleChannelDifference(rd, costCUDA, costTerra);
    static shared_ptr<Texture> jtjDiff = differenceImage(rd, "JTJ Difference", jtjCUDA, jtjTerra);

    jtfDiff->visualization.documentGamma = 2.2f;
    costDiff->visualization.documentGamma = 2.2f;
    jtjDiff->visualization.documentGamma = 2.2f;

    static shared_ptr<Texture> jtjQ = quotientImage(rd, "JTJ Quotient", jtjCUDA, jtjTerra);
    static shared_ptr<Texture> jtfQ = quotientImage(rd, "JTF Quotient", jtfCUDA, jtfTerra);
    static shared_ptr<Texture> costQ = quotientImage(rd, "Cost Quotient", costCUDA, costTerra);
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
    
    String directory = "D:/Projects/DSL/Optimization/Examples/ShapeFromShading/";

    Array<String> filenames;
    FileSystem::getFiles(directory+"*.imagedump", filenames);
    for (auto f : filenames) {
        loadImageDump(directory + f);
    }

    
    
    RenderDevice* rd = RenderDevice::current;
    //compareCUDABlockAndNonBlock(rd, directory);

    compareCUDAAndTerraNonBlock(rd, directory);


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

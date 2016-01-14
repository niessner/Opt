/** \file App.cpp */
#include "App.h"

#include "OpenMesh.h"

#define ARMADILLO 0
#define RAPTOR 1
#define STATUE_HEAD 2
#define MESH_KIND STATUE_HEAD




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

    settings.depthGuardBandThickness    = Vector2int16(0, 0);
    settings.colorGuardBandThickness    = Vector2int16(0, 0);
    settings.dataDir                    = FileSystem::currentDirectory();
    settings.screenshotDirectory        = "../journal/";

    settings.renderer.deferredShading = false;
    settings.renderer.orderIndependentTransparency = false;


    return App(settings).run();
}


App::App(const GApp::Settings& settings) : GApp(settings) {
}

void App::saveMarkerFile() {

    float scale = 1.0f / m_scaleFactor;
    TextOutput to(m_markerFilename);
    to.writeNumber(m_constraints.size());
    for (int i = 0; i < m_constraints.size(); ++i) {
        to.writeNewline();
        to.writeNumber(m_constraints[i].x*scale);
        to.writeNumber(m_constraints[i].y*scale);
        to.writeNumber(m_constraints[i].z*scale);
        to.writeNumber(0.0224524);
        to.writeNumber(m_constraintIndices[i]);
    }
    to.commit();
}

void App::loadMarkerFile() {
    TextInput ti(m_markerFilename);
    int numConstraints = ti.readNumber();//.writeNumber(m_constraints.size());
    m_constraints.resize(numConstraints);
    m_constraintIndices.resize(numConstraints);
    for (int i = 0; i < m_constraints.size(); ++i) {
        //ti.readNewline();
        m_constraints[i].x = ti.readNumber() * m_scaleFactor;
        m_constraints[i].y = ti.readNumber() * m_scaleFactor;
        m_constraints[i].z = ti.readNumber() * m_scaleFactor;
        ti.readNumber();
        m_constraintIndices[i] = ti.readNumber();
    }
    /*
    for (int i = 0; i < m_constraintIndices.size(); ++i) {
        debugPrintf("i: %d\n", i);
        Point3 start = toVec3(m_mesh.point(VertexHandle(m_constraintIndices[i])));
        Point3 newPoint = start * 100.0f;
        if (start.y > 0) {
            if (start.x*100.0f < -40) {
                CFrame frame = CFrame::fromXYZYPRDegrees(-0.45, 0.5, 0.5, 0.0, 0.0, 0.0);
                newPoint = frame.pointToObjectSpace(start) * 100.0f;
            }
            else if (start.x*100.0f > 40) {
                CFrame frame = CFrame::fromXYZYPRDegrees(0.45, 0.5, 0.5, 0.0, 0.0, 0.0);
                newPoint = frame.pointToObjectSpace(start) * 100.0f;
            }
            else {
                CFrame frame = CFrame::fromXYZYPRDegrees(0.0, 0.15, 0.0, 0.0, 30.0, 0.0);
                newPoint = frame.pointToWorldSpace(start) * 100.0f;
            }
        }
        debugPrintf("%f %f %f 0.0224524 %d\n", newPoint.x, newPoint.y, newPoint.z, m_constraintIndices[i]);
    }*/

}


// Called before the application loop begins.  Load data here and
// not in the constructor so that common exceptions will be
// automatically caught.
void App::onInit() {
    GApp::onInit();
    setFrameDuration(1.0f / 60.0f);
    m_modelFrame = CFrame();
# if MESH_KIND == ARMADILLO
    m_scaleFactor = 0.01f;
    m_meshFilename = System::findDataFile("../../../Examples/MeshDeformationARAP/Armadillo20k.ply");
    //m_meshFilename = System::findDataFile("../../../Examples/MeshDeformationARAP/out.ply");
    m_markerFilename = "armadillo.mrk";
    bool doLoadMarkerFile = false;
# elif MESH_KIND == RAPTOR
    m_scaleFactor = 1.0f;
    m_meshFilename = System::findDataFile("../../../Examples/MeshDeformationARAP/raptor_clean.stl");
    m_markerFilename = "raptor.mrk";
    bool doLoadMarkerFile = false;
#else
    m_scaleFactor = 0.02f;
    m_meshFilename = System::findDataFile("../../../Examples/MeshSmoothingLaplacianCOT/serapis.stl");
    m_meshFilename = System::findDataFile("../../../Examples/MeshSmoothingLaplacianCOT/out.off");
    m_markerFilename = "raptor.mrk";
    m_modelFrame = CFrame::fromXYZYPRDegrees(0, 0, 0, -120, -90, 0);
    bool doLoadMarkerFile = false;
#endif
    //m_meshFilename = System::findDataFile("../../../Examples/MeshDeformationED/out.off");

    if (doLoadMarkerFile) {
        loadMarkerFile();
    }


    if (!OpenMesh::IO::read_mesh(m_mesh, m_meshFilename.c_str()))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << m_meshFilename.c_str() << std::endl;
        exit(1);
    }
    m_mesh.update_normals();
    setNewIndex(-1);


    


    // Call setScene(shared_ptr<Scene>()) or setScene(MyScene::create()) to replace
    // the default scene here.
    
    showRenderingStats      = true;

    makeGUI();
    // For higher-quality screenshots:
    developerWindow->videoRecordDialog->setScreenShotFormat("PNG");
    developerWindow->videoRecordDialog->setCaptureGui(false);
    developerWindow->cameraControlWindow->moveTo(Point2(developerWindow->cameraControlWindow->rect().x0(), 0));
    loadScene(
        "Mesh" 
        //developerWindow->sceneEditorWindow->selectedSceneName()  // Load the first scene encountered 
        );
    Any modelAny(Any::TABLE, "ArticulatedModel::Specification");

    modelAny["filename"] = m_meshFilename;
    modelAny["scale"] = m_scaleFactor;
    modelAny["stripMaterials"] = true;
    /*
    Any preprocessAny(STR((
        setMaterial(all(), UniversalMaterial::Specification{
        glossy = Color4(0.6, 0.6, 0.6, 0.5);
        lambertian = Color3(0.9);
        mirrorHint = "STATIC_PROBE";
    }));));
    modelAny["preprocess"] = preprocessAny;*/

    scene()->createModel(modelAny, "meshModel");
    Any entityAny(Any::TABLE, "VisibleEntity");
    entityAny["model"] = "meshModel";
    entityAny["frame"] = m_modelFrame; 
    scene()->createEntity("mesh", entityAny);


}


void App::makeGUI() {
    // Initialize the developer HUD (using the existing scene)
    createDeveloperHUD();
    debugWindow->setVisible(true);
    developerWindow->videoRecordDialog->setEnabled(true);

    GuiPane* infoPane = debugPane->addPane("Info", GuiTheme::ORNATE_PANE_STYLE);
    
    // Example of how to add debugging controls
    infoPane->addButton("Reload Markers", this, &App::loadMarkerFile);
    infoPane->pack();

    // More examples of debugging GUI controls:
    // debugPane->addCheckBox("Use explicit checking", &explicitCheck);
    // debugPane->addTextBox("Name", &myName);
    // debugPane->addNumberBox("height", &height, "m", GuiTheme::LINEAR_SLIDER, 1.0f, 2.5f);
    // button = debugPane->addButton("Run Simulator");

    debugWindow->pack();
    debugWindow->setRect(Rect2D::xywh(0, 0, (float)window()->width(), debugWindow->rect().height()));
}

Vector3 App::toVec3(const Vec3f& v) {
    return Vector3(v[0], v[1], v[2]) * m_scaleFactor;
}

void App::setNewIndex(int index) {
    m_selectedIndex = index;
    m_currentConstraintPosition = toVec3(m_mesh.point(VertexHandle(m_selectedIndex)));

}


static void drawConstraint(RenderDevice* rd, const Point3& p0, const Point3& p1, Color3 c0, Color3 c1) {
    float sphereRadius = 0.01f;
    if ((p0 - p1).length() < 0.01f) {
        c0 = Color3::red();
        c1 = Color3::red();
    }
    Draw::sphere(Sphere(p0, sphereRadius), rd, Color4(c0, 0.5), Color4::clear());
    Draw::sphere(Sphere(p1, sphereRadius), rd, Color4(c1, 0.5), Color4::clear());

    float cylinderRadius = 0.003f;
    Draw::cylinder(Cylinder(p0, p1, cylinderRadius), rd, Color3::black(), Color4::clear());
}

void App::onGraphics3D(RenderDevice* rd, Array<shared_ptr<Surface> >& allSurfaces) {
    if (!scene()) {
        if ((submitToDisplayMode() == SubmitToDisplayMode::MAXIMIZE_THROUGHPUT) && (!rd->swapBuffersAutomatically())) {
            swapBuffers();
        }
        rd->clear();
        rd->pushState(); {
            rd->setProjectionAndCameraMatrix(activeCamera()->projection(), activeCamera()->frame());
            drawDebugShapes();
        } rd->popState();
        return;
    }
    
    screenPrintf("Press R to randomly select a vertex.");
    screenPrintf("Press <- or -> to change through the vertices in order.");
    screenPrintf("Press space to lock in a constraint.");
    screenPrintf("Press enter to save constraints");



    GBuffer::Specification gbufferSpec = m_gbufferSpecification;
    extendGBufferSpecification(gbufferSpec);
    m_gbuffer->setSpecification(gbufferSpec);
    m_gbuffer->resize(m_framebuffer->width(), m_framebuffer->height());
    m_gbuffer->prepare(rd, activeCamera(), 0, -(float)previousSimTimeStep(), m_settings.depthGuardBandThickness, m_settings.colorGuardBandThickness);

    m_renderer->render(rd, m_framebuffer, scene()->lightingEnvironment().ambientOcclusionSettings.enabled ? m_depthPeelFramebuffer : shared_ptr<Framebuffer>(),
        scene()->lightingEnvironment(), m_gbuffer, allSurfaces);

    // Debug visualizations and post-process effects
    rd->pushState(m_framebuffer); {
        // Call to make the App show the output of debugDraw(...)
        rd->setProjectionAndCameraMatrix(activeCamera()->projection(), activeCamera()->frame());
        drawDebugShapes();
        const shared_ptr<Entity>& selectedEntity = (notNull(developerWindow) && notNull(developerWindow->sceneEditorWindow)) ? developerWindow->sceneEditorWindow->selectedEntity() : shared_ptr<Entity>();
        scene()->visualize(rd, selectedEntity, allSurfaces, sceneVisualizationSettings(), activeCamera());
        if (m_selectedIndex >= 0) {
            const Point3& p = toVec3(m_mesh.point(VertexHandle(m_selectedIndex)));
            drawConstraint(rd, p, m_currentConstraintPosition, Color3::red(), Color3::green());
        }
        
        for (int i = 0; i < m_constraintIndices.size(); ++i) {
            drawConstraint(rd,
                toVec3(m_mesh.point(VertexHandle(m_constraintIndices[i]))),
                m_constraints[i], Color3::red(), Color3::purple());
        }

        // Post-process special effects
        m_depthOfField->apply(rd, m_framebuffer->texture(0), m_framebuffer->texture(Framebuffer::DEPTH), activeCamera(), m_settings.depthGuardBandThickness - m_settings.colorGuardBandThickness);

        m_motionBlur->apply(rd, m_framebuffer->texture(0), m_gbuffer->texture(GBuffer::Field::SS_EXPRESSIVE_MOTION),
            m_framebuffer->texture(Framebuffer::DEPTH), activeCamera(),
            m_settings.depthGuardBandThickness - m_settings.colorGuardBandThickness);
    } rd->popState();

    // We're about to render to the actual back buffer, so swap the buffers now.
    // This call also allows the screenshot and video recording to capture the
    // previous frame just before it is displayed.
    if (submitToDisplayMode() == SubmitToDisplayMode::MAXIMIZE_THROUGHPUT) {
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

    if (ui->keyDown(GKey::LCTRL) || ui->keyDown(GKey::RCTRL)) {
        float s = 0.005f;
        if (ui->keyDown(GKey('w')) || ui->keyDown(GKey('i'))) {
            m_currentConstraintPosition += Vector3(0, 0, -s);
        }
        if (ui->keyDown(GKey('s')) || ui->keyDown(GKey('k'))) {
            m_currentConstraintPosition += Vector3(0, 0, s);
        }
        if (ui->keyDown(GKey('a')) || ui->keyDown(GKey('j'))) {
            m_currentConstraintPosition += Vector3(-s, 0, 0);
        }
        if (ui->keyDown(GKey('d')) || ui->keyDown(GKey('l'))) {
            m_currentConstraintPosition += Vector3(s, 0, 0);
        }

        if (ui->keyDown(GKey('q')) || ui->keyDown(GKey('u'))) {
            m_currentConstraintPosition += Vector3(0, -s, 0);
        }
        if (ui->keyDown(GKey('e')) || ui->keyDown(GKey('o'))) {
            m_currentConstraintPosition += Vector3(0, s, 0);
        }
        if (ui->keyPressed(GKey('v'))) {

            const Ray& eyeRay = scene()->eyeRay(activeCamera(), ui->mouseXY(), RenderDevice::current->viewport(), Vector2int16(0, 0));
            float distance = finf();
            scene()->intersect(eyeRay, distance);
            Point3 hitPoint = eyeRay.origin() + eyeRay.direction() * distance;

            float closestDistance = finf();
            int newIndex = m_selectedIndex;
            for (int i = 0; i < m_mesh.n_vertices(); ++i) {
                float newDist = (toVec3(m_mesh.point(VertexHandle(i))) - hitPoint).squaredLength();
                if (newDist < closestDistance) {
                    newIndex = i;
                    closestDistance = newDist;
                }
            }
            setNewIndex(newIndex);

        }
    }


    GApp::onUserInput(ui);

    if (ui->keyPressed(GKey('r'))) {
        setNewIndex(Random::common().integer(0, m_mesh.n_vertices() - 1));
    }

    if (ui->keyPressed(GKey::LEFT)) {
        setNewIndex((m_selectedIndex + m_mesh.n_vertices() - 1) % m_mesh.n_vertices());
    }
    if (ui->keyPressed(GKey::RIGHT)) {
        setNewIndex((m_selectedIndex + 1) % m_mesh.n_vertices());
    }
    if (ui->keyPressed(GKey::RETURN)) {
        saveMarkerFile();
    }
    if (ui->keyPressed(GKey::SPACE)) {
        m_constraints.append(m_currentConstraintPosition);
        m_constraintIndices.append(m_selectedIndex);
    }



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

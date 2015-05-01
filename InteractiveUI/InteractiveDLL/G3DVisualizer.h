#ifndef G3DVisualizer_h
#define G3DVisualizer_h

#ifdef KEY_DOWN
#undef KEY_DOWN
#endif

#ifdef KEY_UP
#undef KEY_UP
#endif
#include <G3D/G3DAll.h>
#include <mutex>
#include "Optimizer.h"
class G3DVisualizer : public GApp {
private:
    /** Signal that we can receive events */
    bool m_initialized;

    String m_currentFilename;

    int m_currentInputIndex;

    int m_outputWidth;
    int m_outputHeight;
    int m_outputChannelCount;

    bool m_repositionVisualizationNextFrame;

    GuiDropDownList* m_inputDropDownList;


    Optimizer m_optimizer;

    
    OptimizationTimingInfo m_optimizationTiming;
    std::mutex             m_oTimingMutex;

    struct RunOptMessage {
        std::mutex  mutex;
        std::string terraFilename;
        std::string optimizationMethod;
        RealTime    lastMessageTime;
    };
    RunOptMessage   m_lastRunMessage;

    RealTime m_lastOptimizationRunTime;


    Array<shared_ptr<TextureBrowserWindow>> m_textureBrowserWindows;
    
    void getRunOptMessage(std::string& terraFilename, std::string& optMethod, RealTime& lastMessageTime) {
        m_lastRunMessage.mutex.lock(); {
            lastMessageTime = m_lastRunMessage.lastMessageTime;
            optMethod       = m_lastRunMessage.optimizationMethod;
            terraFilename   = m_lastRunMessage.terraFilename;
        } m_lastRunMessage.mutex.unlock();
    }

    


    void writeTimingInfo(const OptimizationTimingInfo& t) {
        m_oTimingMutex.lock(); {
            m_optimizationTiming = t;
        } m_oTimingMutex.unlock();
    }

    Array<OptimizationInput>    m_inputs;
    OptimizationOutput          m_output;

    G3D_DECLARE_ENUM_CLASS(VisualizationMode, IMAGE, POINT_CLOUD);
    VisualizationMode m_outputVisualizationMode;

    /** Called from onInit */
    void makeGUI();

    struct MoveMessage {
        std::mutex  mutex;
        bool fresh;
        int width;
        int height;
        int x;
        int y;
        MoveMessage() : fresh(false) {}
    };
    MoveMessage m_moveMessage;


    void repositionVisualizations();

public:
    void sendMoveMessage(int x, int y, int width, int height) {
        m_moveMessage.mutex.lock(); {
            m_moveMessage.x = x;
            m_moveMessage.y = y;
            m_moveMessage.width = width;
            m_moveMessage.height = height;
            m_moveMessage.fresh = true;
        } m_moveMessage.mutex.unlock();
    }
    void executeMoveMessage() {
        m_moveMessage.mutex.lock(); {
            if (m_moveMessage.fresh) {
                moveWindow(m_moveMessage.x, m_moveMessage.y, m_moveMessage.width, m_moveMessage.height);
            }
            m_moveMessage.fresh = false;
        } m_moveMessage.mutex.unlock();
    }

    void sendRunOptMessage(const std::string& terraFilename, const std::string& optMethod) {
        m_lastRunMessage.mutex.lock(); {
            m_lastRunMessage.lastMessageTime = System::time();
            m_lastRunMessage.optimizationMethod = optMethod;
            m_lastRunMessage.terraFilename = terraFilename;
        } m_lastRunMessage.mutex.unlock();
    }

    OptimizationTimingInfo acquireTimingInfo() {
        OptimizationTimingInfo copy;
        m_oTimingMutex.lock(); {
            copy = m_optimizationTiming;
        } m_oTimingMutex.unlock();
        return copy;
    }

    void loadNewInput();

    void loadInput(const String& filename, int inputIndex);

    void loadImage(const String& filename, int inputIndex);

    void loadDepthColorFrame(const String& filename, int inputIndex);

    bool initialized() const {
        return m_initialized;
    }

    G3DVisualizer(const GApp::Settings& settings = GApp::Settings());

    virtual void onInit();
    virtual void onGraphics3D(RenderDevice* rd, Array< shared_ptr<Surface> >& surface);
    virtual void onGraphics2D(RenderDevice* rd, Array< shared_ptr<Surface2D> >& surface2D);
    virtual void onSimulation(RealTime rdt, SimTime sdt, SimTime idt);

    virtual bool onEvent(const GEvent& e);

    void moveWindow(int x, int y, int width, int height);

};

#endif
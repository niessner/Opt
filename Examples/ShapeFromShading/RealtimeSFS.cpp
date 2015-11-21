
#include "stdafx.h"

#include "RealtimeSFS.h"
#include "CUDARGBDSensor.h"
#include "Cuda/CUDACameraTrackingMultiRes.h"
#include "StdOutputLogger.h"
#include "Timer.h"
#include "DX11QuadDrawer.h"
#include "DXUT.h"
#include "GlobalAppState.h"
#include "KinectOneSensor.h"
#include "PrimeSenseSensor.h"
#include "GroundTruthSensor.h"
#include "BinaryDumpReader.h"
#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "GlobalAppState.h"
#include "DX11PhongLighting.h"
#include "TimingLog.h"
#include "GlobalCameraTrackingState.h"
#include "CUDARGBDAdapter.h"
#include "DX11RGBDRenderer.h"
#include "DX11CustomRenderTarget.h"

#ifdef KINECT_ONE
#pragma comment(lib, "Kinect20.lib")
#endif

#ifdef OPEN_NI
#pragma comment(lib, "OpenNI2.lib")
#endif

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

RGBDSensor* getRGBDSensor()
{
#ifdef OPEN_NI
	if (GlobalAppState::get().s_sensorIdx == 1)
	{
		static PrimeSenseSensor s_primeSense;	//TODO make parameters consistent
		return &s_primeSense;
	}
#endif
#ifdef KINECT_ONE
	if (GlobalAppState::get().s_sensorIdx == 2)
	{
		static KinectOneSensor s_kinect;	//TODO make parameters consistent
		return &s_kinect;
	} 
#endif
#ifdef GROUNDTRUTH_SENSOR
	if (GlobalAppState::get().s_sensorIdx == 3) {
		static GroundTruthSensor s_groundtruth;	//TODO make parameters consistent
		return &s_groundtruth;
	} 
#endif
#ifdef BINARY_DUMP_READER
	if (GlobalAppState::get().s_sensorIdx == 4) {
		static BinaryDumpReader s_binaryDump;
		return &s_binaryDump;
	}
#endif
	
	throw MLIB_EXCEPTION("unkown sensor id " + std::to_string(GlobalAppState::get().s_sensorIdx));
	
	return NULL;
}

CModelViewerCamera          g_Camera;               // A model viewing camera
CUDARGBDSensor				g_KinectSensor;
CUDARGBDAdapter				g_RGBDAdapter;
DX11RGBDRenderer			g_RGBDRenderer;
DX11CustomRenderTarget		g_CustomRenderTarget;
DX11CustomRenderTarget		g_RenderToFileTarget;

bool g_deviceCreated = false;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4
#define IDC_TEST                  5

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK		ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK		OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
LRESULT CALLBACK	MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,	void* pUserContext );
void CALLBACK		OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void CALLBACK		OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
bool CALLBACK		IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK	OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
HRESULT CALLBACK	OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,	const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK		OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK		OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK		OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext );


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif


	try {
		std::string fileNameDescGlobalApp;
		std::string fileNameDescGlobalTracking;
		if (argc == 3) {
			fileNameDescGlobalApp = std::string(argv[1]);
			fileNameDescGlobalTracking = std::string(argv[2]);
		} else {
			std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking]" << std::endl;
			fileNameDescGlobalApp = "zParametersDefault.txt";
			fileNameDescGlobalTracking = "zParametersTrackingDefault.txt";
		}
		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalTracking) << " = " << fileNameDescGlobalTracking << std::endl;

		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);
		GlobalAppState::getInstance().print();
		
		// Set DXUT callbacks
		DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
		DXUTSetCallbackMsgProc( MsgProc );
		DXUTSetCallbackKeyboard( OnKeyboard );
		DXUTSetCallbackFrameMove( OnFrameMove );

		DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
		DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
		DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
		DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
		DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
		DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );
				
		DXUTInit( true, true ); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
		DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen		
		DXUTCreateWindow( GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"RealtimeSFS", true);

		DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
		DXUTMainLoop(); // Enter into the DXUT render loop

	}
	catch(const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch(...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	// For the first device created if its a REF device, optionally display a warning dialog box
	static bool s_bFirstTime = true;
	if( s_bFirstTime )
	{
		s_bFirstTime = false;
		if( ( DXUT_D3D9_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF ) ||
			( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
			pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
		{
			DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
		}
	}

	return true;
}

//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	g_Camera.FrameMove( fElapsedTime );
	// Update the camera's position based on user input 
}

//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
	void* pUserContext )
{

	 g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}

//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
static int whichShot = 0;
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	GlobalAppState::get().s_bRenderModeChanged = true;

	if( bKeyDown ) {
		wchar_t sz[200];

		switch( nChar )
		{
		case VK_F1:
			swprintf_s(sz, 200, L"screenshot%d.bmp", whichShot++);
			DXUTSnapD3D11Screenshot(sz, D3DX11_IFF_BMP);
			std::wcout << std::wstring(sz) << std::endl;
			break;
		case '1':
			GlobalAppState::get().s_RenderMode = 1;
			break;
		case '2':
			GlobalAppState::get().s_RenderMode = 2;
			break;
		case '3':
			GlobalAppState::get().s_RenderMode = 3;
			break;
		case '4':
			GlobalAppState::get().s_RenderMode = 4;
			break;
		case '5':
			GlobalAppState::get().s_RenderMode = 5;
			break;
		case '6':
			GlobalAppState::get().s_RenderMode = 6;
			break;
		case '7':
			GlobalAppState::get().s_RenderMode = 7;
			break;
		case '8':
			GlobalAppState::get().s_RenderMode = 8;
			break;
		case '9':
			GlobalAppState::get().s_RenderMode = 9;
			break;
		case '0':
			GlobalAppState::get().s_RenderMode = 0;
			break;
		case 'T':
			GlobalAppState::get().s_timingsDetailledEnabled = !GlobalAppState::get().s_timingsDetailledEnabled;
			break;
		case 'Z':
			GlobalAppState::get().s_timingsTotalEnabled = !GlobalAppState::get().s_timingsTotalEnabled;
			break;
		case 'H':
			g_KinectSensor.toggleFillHoles();
			break;
		case 'S':
			g_KinectSensor.toggleSDSRefinement();
			break;
		case 'F':
			g_KinectSensor.saveCameraSpaceMapAsMesh(GlobalAppState::get().s_onlySaveForeground);
			break;
		case 'M':
			GlobalAppState::get().s_renderToFile = !GlobalAppState::get().s_renderToFile;
			break;
        case 'B':
            GlobalAppState::get().s_useBlockSolver = !GlobalAppState::get().s_useBlockSolver;
            if (CUDAPatchSolverSFS::s_optimizerAD != NULL) {
                g_KinectSensor.resetOptPlans();
                delete CUDAPatchSolverSFS::s_optimizerAD;
                CUDAPatchSolverSFS::s_optimizerAD = NULL;
            }
            if (CUDAPatchSolverSFS::s_optimizerNoAD != NULL) {
                g_KinectSensor.resetOptPlans();
                delete CUDAPatchSolverSFS::s_optimizerNoAD;
                CUDAPatchSolverSFS::s_optimizerNoAD = NULL;
            }
            break;
        case 'O':
            GlobalAppState::get().s_optimizer = (GlobalAppState::get().s_optimizer + 1) % 3;
            break;
        case 'G':
            for (int i = 0; i < GlobalAppState::get().s_weightShadingStart.size(); ++i) {
                GlobalAppState::get().s_weightShadingStart[i] = GlobalAppState::get().s_weightShadingStart[i] * 2.0f;
            }     
            break;
		case ' ':
			GlobalAppState::get().s_playData = !GlobalAppState::get().s_playData;
			break;
		case 'P':
			{
				if (GlobalAppState::get().s_bRecordDepthData == true) {
					std::cout << "saving recorded data to file..." << std::endl;
					g_KinectSensor.saveRecordedFramesToFile();
					std::cout << "stopped recording: press 'P' to start again" << std::endl;
					GlobalAppState::get().s_bRecordDepthData = false;
				} else {
					GlobalAppState::get().s_bRecordDepthData = true;
					std::cout << "started recording: 'P' press p to stop and save to file" << std::endl;
				}
				break;
			}
		case 'U':
		{
			if(GlobalAppState::get().s_convergenceAnalysisIsRunning)
			{
				GlobalAppState::get().s_convergenceAnalysis.saveGraph("result.conv");
				GlobalAppState::get().s_convergenceAnalysis.reset();
				std::cout << "saving convergence to file..." << std::endl;
			}
			else
			{
				std::cout << "starting convergence analysis..." << std::endl;
			}

			GlobalAppState::get().s_convergenceAnalysisIsRunning = !GlobalAppState::get().s_convergenceAnalysisIsRunning;
			
			break;
		}
		case 'E':
			g_KinectSensor.saveDepthAndColor();
			break;
		case 'R':
			g_KinectSensor.reset();
			std::cout << "reset" << std::endl;
			break;
		case VK_F3:
			GlobalAppState::get().s_texture_threshold += 0.02;
			std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			if(GlobalAppState::get().s_texture_threshold>1.0f)
				GlobalAppState::get().s_texture_threshold = 1.0f;
			break;
		case VK_F4:
			GlobalAppState::get().s_texture_threshold -= 0.02;
			std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			if(GlobalAppState::get().s_texture_threshold<0.0f)
				GlobalAppState::get().s_texture_threshold = 0.0f;
			break;
        case VK_F5:
            if (CUDAPatchSolverSFS::s_optimizerAD != NULL) {
                g_KinectSensor.resetOptPlans();
                delete CUDAPatchSolverSFS::s_optimizerAD;
                CUDAPatchSolverSFS::s_optimizerAD = NULL;
            }
            if (CUDAPatchSolverSFS::s_optimizerNoAD != NULL) {
                g_KinectSensor.resetOptPlans();
                delete CUDAPatchSolverSFS::s_optimizerNoAD;
                CUDAPatchSolverSFS::s_optimizerNoAD = NULL;
            }
            break;
		default:
			break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	GlobalAppState::get().s_bRenderModeChanged = true;

	switch( nControlID )
	{
		// Standard DXUT controls
		case IDC_TOGGLEFULLSCREEN:
			DXUTToggleFullScreen(); 
			break;
		case IDC_TOGGLEREF:
			DXUTToggleREF(); 
			break;
		case IDC_TEST:
			break;
	}
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
	return true;
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	HRESULT hr = S_OK;
	
	V_RETURN(GlobalAppState::get().OnD3D11CreateDevice(pd3dDevice));

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
	
	if(getRGBDSensor() == NULL)
	{
		std::cout << "No RGBD Sensor specified" << std::endl;
		while(1);
	}

	if ( FAILED( getRGBDSensor()->createFirstConnected() ) )
	{
		MessageBox(NULL, L"No ready Depth Sensor found!", L"Error", MB_ICONHAND | MB_OK);
		return S_FALSE;
	}

	//static init
	V_RETURN(g_RGBDAdapter.OnD3D11CreateDevice(pd3dDevice, getRGBDSensor(), GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight)); 
	V_RETURN(g_KinectSensor.OnD3D11CreateDevice(pd3dDevice, &g_RGBDAdapter)); 

	V_RETURN(DX11QuadDrawer::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11PhongLighting::OnD3D11CreateDevice(pd3dDevice));
	
	TimingLog::init();

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_adapterWidth, GlobalAppState::get().s_adapterHeight, formats));

	std::vector<DXGI_FORMAT> rtfFormat;
	rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
	V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_renderToFileResWidth, GlobalAppState::get().s_renderToFileResHeight, rtfFormat));

	g_deviceCreated = true;

	D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	g_Camera.SetViewParams( &vecEye, &vecAt );

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	DXUTGetGlobalResourceCache().OnDestroyDevice();

	DX11QuadDrawer::OnD3D11DestroyDevice();
	DX11PhongLighting::OnD3D11DestroyDevice();
	GlobalAppState::get().OnD3D11DestroyDevice();

	g_KinectSensor.OnD3D11DestroyDevice();
	g_RGBDAdapter.OnD3D11DestroyDevice();

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();
	g_RenderToFileTarget.OnD3D11DestroyDevice();

	TimingLog::destroy();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
	const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr = S_OK;


	// Setup the camera's projection parameters
	g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );
	//g_Camera.SetRotateButtons(true, false, false);

	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	//D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	//D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	//g_Camera.SetViewParams( &vecEye, &vecAt );
	g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.1f, 10.0f );


	V_RETURN(DX11PhongLighting::OnResize(pd3dDevice, GlobalAppState::get().s_windowWidth = pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height));

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------

void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext )
{
	
	if(!g_deviceCreated) return;

	if(GlobalAppState::get().s_bRenderModeChanged)
	{
		GlobalAppState::get().s_bRenderModeChanged = false;
	}

	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	//if(g_D3DSettingsDlg.IsActive())
	//{
	//	g_D3DSettingsDlg.OnRender(fElapsedTime);
	//	return;
	//}

	// Clear the back buffer
	static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
	pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	// if we have received any valid new depth data we may need to draw
	HRESULT hr0 = g_KinectSensor.process(pd3dImmediateContext);

	// Filtering
	g_KinectSensor.setFiterDepthValues(GlobalAppState::get().s_depthFilter, GlobalAppState::get().s_depthSigmaD, GlobalAppState::get().s_depthSigmaR);
	g_KinectSensor.setFiterIntensityValues(GlobalAppState::get().s_colorFilter, GlobalAppState::get().s_colorSigmaD, GlobalAppState::get().s_colorSigmaR);

	HRESULT hr = S_OK;

	///////////////////////////////////////
	// Render
	///////////////////////////////////////

	//Start Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.start(); }

		if(GlobalAppState::get().s_RenderMode == 1)
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapRefinedFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);

			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		}
		else if(GlobalAppState::get().s_RenderMode == 2)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetCameraSpacePositionsFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
		}
		else if(GlobalAppState::get().s_RenderMode == 3)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetColorMapFilteredFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
		}
		else if(GlobalAppState::get().s_RenderMode == 4)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetNormalMapFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
		}
		else if(GlobalAppState::get().s_RenderMode == 5)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetDepthMapRefinedFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());	
		}
		else if(GlobalAppState::get().s_RenderMode == 6)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetAlbedoFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
		}
		else if(GlobalAppState::get().s_RenderMode == 7)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetIntensityMapFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());			
		}
		else if(GlobalAppState::get().s_RenderMode == 8)
		{
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetShadingFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());			
		}
		else if(GlobalAppState::get().s_RenderMode == 9)
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapColorSpaceFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);

			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
		}
		else if(GlobalAppState::get().s_RenderMode == 0)
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapRawFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);
			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);

			//DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetFilteredDepthMapMaskFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
		}

	// Stop Timing
	if(GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.stop(); TimingLog::totalTimeRender += GlobalAppState::get().s_Timer.getElapsedTimeMS(); TimingLog::countTimeRender++; }

	if (hr0 == S_OK && GlobalAppState::get().s_playData && GlobalAppState::get().s_renderToFile && g_RGBDAdapter.getFrameNumber() > 1) {
		unsigned int frameNumber = g_RGBDAdapter.getFrameNumber() - 2;
		std::string baseFolder = "output\\";
		CreateDirectory(L"output", NULL);
		CreateDirectory(L"output\\rawPhong", NULL);
		CreateDirectory(L"output\\refinedPhong", NULL);
		CreateDirectory(L"output\\smoothPhong", NULL);
		CreateDirectory(L"output\\color", NULL);
		CreateDirectory(L"output\\normals", NULL);
		CreateDirectory(L"output\\normalsRefined", NULL);
		CreateDirectory(L"output\\albedo", NULL);
		CreateDirectory(L"output\\shading", NULL);
		CreateDirectory(L"output\\intensity", NULL);
		CreateDirectory(L"output\\depth", NULL);

		std::stringstream ssFrameNumber;	unsigned int numCountDigits = 6;
		for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)frameNumber+1))); i < numCountDigits; i++) ssFrameNumber << "0";
		ssFrameNumber << frameNumber;

		{
			//depth
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetCameraSpacePositionsFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "depth\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			//albedo
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetAlbedoFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "albedo\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			//intensity
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetIntensityMapFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());			
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "intensity\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}

		{
			//shading
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetShadingFloat(), 1, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());			
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "shading\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}

		{
			//color
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_RGBDAdapter.GetColorMapResampledFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "color\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.getNormalMapNoRefinementFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "normals\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuadDynamic(DXUTGetD3D11Device(), pd3dImmediateContext, (float*)g_KinectSensor.GetNormalMapFloat4(), 4, g_KinectSensor.getColorWidth(), g_KinectSensor.getColorHeight());
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "normalsRefined\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapRefinedFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);

			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "refinedPhong\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapRawFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);
			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "rawPhong\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
		{
			const mat4f renderIntrinsics = g_RGBDAdapter.getColorIntrinsics();
			const mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

			g_CustomRenderTarget.Clear(pd3dImmediateContext);
			g_CustomRenderTarget.Bind(pd3dImmediateContext);
			g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_KinectSensor.GetDepthMapColorSpaceFloat(), g_KinectSensor.GetColorMapFilteredFloat4(), g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), g_RGBDAdapter.getColorIntrinsicsInv(), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
			g_CustomRenderTarget.Unbind(pd3dImmediateContext);
			DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());

			g_RenderToFileTarget.Clear(pd3dImmediateContext);
			g_RenderToFileTarget.Bind(pd3dImmediateContext);
			DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
			g_RenderToFileTarget.Unbind(pd3dImmediateContext);

			BYTE* data; unsigned int bytesPerElement;
			g_RenderToFileTarget.copyToHost(data, bytesPerElement);
			ColorImageR8G8B8A8 image(g_RenderToFileTarget.getHeight(), g_RenderToFileTarget.getWidth(), (vec4uc*)data);
			for (unsigned int i = 0; i < image.getWidth()*image.getHeight(); i++) {
				if (image.getDataPointer()[i].x > 0 || image.getDataPointer()[i].y > 0 || image.getDataPointer()[i].z > 0)
					image.getDataPointer()[i].w = 255;
			}
			FreeImageWrapper::saveImage(baseFolder + "smoothPhong\\" + ssFrameNumber.str() + ".png", image);
			SAFE_DELETE_ARRAY(data);
		}
	}

	TimingLog::printTimings();

	DXUT_EndPerfEvent();
}

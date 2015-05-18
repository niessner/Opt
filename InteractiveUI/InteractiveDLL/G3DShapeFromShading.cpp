#include "G3DShapeFromShading.h"
#include "SFSHelpers.h"
#include "cuda_SimpleMatrixUtil.h"
#include "mLibCore.h"
#include "CudaImage.h"
#include "PatchSolverSFS/CUDAPatchSolverSFS.h"
using G3D::PixelTransferBuffer;
using G3D::GLPixelTransferBuffer;
using G3D::ImageFormat;
using G3D::Vector2int32;

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height);
extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void convertDepthRawToFloat(float* d_output, unsigned short* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);
extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height);

extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, float* d_depthMaskMap);
extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);


void G3DShapeFromShading::resampleImages(shared_ptr<Texture> inputDepth, shared_ptr<Texture> inputColor, shared_ptr<Texture> outputDepth, shared_ptr<Texture> outputColor) {
    alwaysAssertM(outputDepth->width() == outputColor->width() && outputDepth->height() == outputColor->height(), "Target texture must be same size");
    Vector2int32 targetDim(outputDepth->width(), outputDepth->height());
    ////////////////////////////////////////////////////////////////////////////////////
    // Process Color
    ////////////////////////////////////////////////////////////////////////////////////

    shared_ptr<CudaImage> cudaColorMapRaw = CudaImage::fromTexture(inputColor);
    shared_ptr<CudaImage> colorMapFloat = CudaImage::createEmpty(inputColor->width(), inputColor->height(), ImageFormat::RGBA32F());
    shared_ptr<CudaImage> colorMapResampled = CudaImage::createEmpty(targetDim.x, targetDim.y, ImageFormat::RGBA32F());
    convertColorRawToFloat4((float4*)colorMapFloat->data(), (BYTE*)cudaColorMapRaw->data(), inputColor->width(), inputColor->height());
    cutilSafeCall(cudaDeviceSynchronize());
    if ((inputColor->width() == targetDim.x) && (inputColor->height() == targetDim.y)) copyFloat4Map((float4*)colorMapResampled->data(), (float4*)colorMapFloat->data(), targetDim.x, targetDim.y);
    else																						   resampleFloat4Map((float4*)colorMapResampled->data(), targetDim.x, targetDim.y, (float4*)colorMapFloat->data(), inputColor->width(), inputColor->height());
    cutilSafeCall(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////////////////////////////
    // Process Depth
    ////////////////////////////////////////////////////////////////////////////////////
    shared_ptr<CudaImage> depthMapFloat = CudaImage::fromTexture(inputDepth);
    shared_ptr<CudaImage> depthMapResampled = CudaImage::createEmpty(targetDim.x, targetDim.y, ImageFormat::R32F());
    cutilSafeCall(cudaDeviceSynchronize());
    resampleFloatMap((float*)depthMapResampled->data(), targetDim.x, targetDim.y, (float*)depthMapFloat->data(), inputDepth->width(), inputDepth->height(), NULL);
    cutilSafeCall(cudaDeviceSynchronize());
    depthMapResampled->updateTexture(outputDepth);
    colorMapResampled->updateTexture(outputColor);
    cutilSafeCall(cudaDeviceSynchronize());
}

void G3DShapeFromShading::estimateLightingAndAlbedo(shared_ptr<Texture> color, shared_ptr<Texture> depth, shared_ptr<Texture> outputAlbedo, shared_ptr<Texture> targetLuminance, shared_ptr<Texture> albedoLuminance, G3D::Array<float>& lightinSHCoefficients) {
    if (G3D::isNull(m_sfsHelpers)) {
        m_sfsHelpers = shared_ptr<SFSHelpers>(new SFSHelpers());
    }
    int width = color->width();
    int height = color->height();

    static shared_ptr<Texture> targetColor      = Texture::createEmpty("SFSTargetColor", width, height, ImageFormat::RGBA32F());
    Texture::copy(color, targetColor);

    static shared_ptr<Texture> normals = Texture::createEmpty("SFSNormalEstimate", width, height, ImageFormat::RGBA32F());

    shared_ptr<CudaImage> cudaDepthImage = CudaImage::fromTexture(depth);
    shared_ptr<CudaImage> cudaCSPosition    = CudaImage::createEmpty(width, height, ImageFormat::RGBA32F());
    shared_ptr<CudaImage> cudaTargetColor = CudaImage::fromTexture(targetColor);
    shared_ptr<CudaImage> cudaTargetIntensity = CudaImage::fromTexture(targetLuminance);

    float horizontalFieldOfView = 1.082104;
    float verticalFieldOfView = 0.848230;
    float centerX = (width / 2.0f);
    float centerY = (height / 2.0f);
    float focalLengthX = centerX / tan(horizontalFieldOfView / 2.0f);
    float focalLengthY = centerY / tan(verticalFieldOfView / 2.0f);
    ml::mat4f m_colorIntrinsicsInv(
        1.0f / focalLengthX, 0.0f, 0.0f, -centerX*1.0f / focalLengthX,
        0.0f, -1.0f / focalLengthY, 0.0f, centerY*1.0f / focalLengthY,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

   
    // For writing out unrefined mesh/and ICP
    float4x4 M(m_colorIntrinsicsInv.getPointer());
    convertDepthFloatToCameraSpaceFloat4((float4*)cudaCSPosition->data(), (float*)cudaDepthImage->data(), M, width, height);

    
    //cudaDepthImage->updateTexture(depthTestReadback);



    ///////////////////////////////////////////////////////////////////////////////////
    //// Camera Tracking
    ///////////////////////////////////////////////////////////////////////////////////	

    shared_ptr<CudaImage> cudaNormals = CudaImage::createEmpty(width, height, ImageFormat::RGBA32F());
    shared_ptr<CudaImage> cudaAlbedo = CudaImage::createEmpty(width, height, ImageFormat::RGBA32F());
    shared_ptr<CudaImage> cudaAlbedoLuminance = CudaImage::fromTexture(albedoLuminance);
    computeNormals((float4*)cudaNormals->data(), (float4*)cudaCSPosition->data(), width, height);
    
    convertColorToIntensityFloat((float*)cudaTargetIntensity->data(), (float4*)cudaTargetColor->data(), width, height);

    shared_ptr<CudaImage> lightingCoefficients = CudaImage::createEmpty(9, 1, ImageFormat::R32F());
    float thres_depth = 1.0f;
    m_sfsHelpers->solveLighting((float*)cudaDepthImage->data(), (float*)cudaTargetIntensity->data(), (float4*)cudaNormals->data(), NULL, (float*)lightingCoefficients->data(), thres_depth, color->width(), color->height());
    m_sfsHelpers->solveReflectance((float*)cudaDepthImage->data(), (float4*)cudaTargetColor->data(), (float4*)cudaNormals->data(), (float*)lightingCoefficients->data(), (float4*)cudaAlbedo->data(), width, height);

    convertColorToIntensityFloat((float*)cudaAlbedoLuminance->data(), (float4*)cudaAlbedo->data(), width, height);
    cudaAlbedoLuminance->updateTexture(albedoLuminance);
    cudaTargetIntensity->updateTexture(targetLuminance);
    cudaAlbedo->updateTexture(outputAlbedo);
    cudaNormals->updateTexture(normals);

    float coeffs[9];
    CUDA_SAFE_CALL(cudaMemcpy(coeffs, lightingCoefficients->data(), sizeof(float) * 9, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 9; ++i) {
        G3D::debugPrintf("L[%d] = %f\n", i, coeffs[i]);
    }
}

void G3DShapeFromShading::solveSFS(shared_ptr<Texture> result, shared_ptr<Texture> targetDepth, shared_ptr<Texture> outputAlbedo, shared_ptr<Texture> targetLuminance, shared_ptr<Texture> albedoLuminance, const G3D::Array<float>& lightingSHCoefficients) {
   /* Matrix4f M(colorIntrinsics.getPointer()); M.transposeInPlace();	//TODO check!
    CUDAPatchSolverSFS* patchSolver = new CUDAPatchSolverSFS(M, result->width(), result->height(), 0);




    patchSolver->solveSFS(d_depthMapColorSpaceFloat, d_depthMapRefinedLastFrameFloat, d_depthMapMaskMorphFloat, d_intensityMapFloat, d_maskEdgeMapUchar, deltaTransform, GlobalAppState::get().s_nNonLinearIterations, GlobalAppState::get().s_nLinearIterations, GlobalAppState::get().s_nPatchIterations, GlobalAppState::get().s_weightFitting, GlobalAppState::get().s_weightShadingIncrement, GlobalAppState::get().s_weightShadingStart, GlobalAppState::get().s_weightBoundary, GlobalAppState::get().s_weightRegularizer, GlobalAppState::get().s_weightPrior, d_lightingCoeffFloat, NULL, d_depthMapRefinedFloat, GlobalAppState::get().s_refineForeground);
*/
}
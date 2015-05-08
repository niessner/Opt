#include "G3DShapeFromShading.h"
#include "SFSHelpers.h"
#include "cuda_SimpleMatrixUtil.h"
#include "mLibCore.h"
#include "CudaImage.h"
using G3D::PixelTransferBuffer;
using G3D::GLPixelTransferBuffer;
using G3D::ImageFormat;

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height);
extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);

void G3DShapeFromShading::estimateLightingAndAlbedo(shared_ptr<Texture> color, shared_ptr<Texture> depth, shared_ptr<Texture> outputAlbedo, Array<float>& lightinSHCoefficients) {
    if (G3D::isNull(m_sfsHelpers)) {
        m_sfsHelpers = shared_ptr<SFSHelpers>(new SFSHelpers());
    }
    int width = color->width();
    int height = color->height();

    static shared_ptr<Texture> targetIntensity  = Texture::createEmpty("SFSTargetIntensity", width, height, ImageFormat::R32F());
    static shared_ptr<Texture> targetColor      = Texture::createEmpty("SFSTargetColor", width, height, ImageFormat::RGBA32F());
    Texture::copy(color, targetColor);

    shared_ptr<CudaImage> cudaDepthImage = CudaImage::fromTexture(depth);
    shared_ptr<CudaImage> cudaCSPosition    = CudaImage::createEmpty(width, height, ImageFormat::RGBA32F());
    shared_ptr<CudaImage> cudaTargetColor = CudaImage::fromTexture(targetColor);
    shared_ptr<CudaImage> cudaTargetIntensity = CudaImage::createEmpty(width, height, ImageFormat::R32F());

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

    computeNormals((float4*)cudaNormals->data(), (float4*)cudaCSPosition->data(), width, height);
    
    convertColorToIntensityFloat((float*)cudaTargetIntensity->data(), (float4*)cudaTargetColor->data(), width, height);

    shared_ptr<CudaImage> lightingCoefficients = CudaImage::createEmpty(9, 1, ImageFormat::R32F());
    float thres_depth = 1.0f;
    m_sfsHelpers->solveLighting((float*)cudaDepthImage->data(), (float*)cudaTargetIntensity->data(), (float4*)cudaNormals->data(), NULL, (float*)lightingCoefficients->data(), thres_depth, color->width(), color->height());
    m_sfsHelpers->solveReflectance((float*)cudaDepthImage->data(), (float4*)cudaTargetColor->data(), (float4*)cudaNormals->data(), (float*)lightingCoefficients->data(), (float4*)cudaAlbedo->data(), width, height);

    cudaTargetIntensity->updateTexture(targetIntensity);
    cudaAlbedo->updateTexture(outputAlbedo);

    float coeffs[9];
    CUDA_SAFE_CALL(cudaMemcpy(coeffs, lightingCoefficients->data(), sizeof(float) * 9, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 9; ++i) {
        G3D::debugPrintf("L[%d] = %f\n", i, coeffs[i]);
    }
}
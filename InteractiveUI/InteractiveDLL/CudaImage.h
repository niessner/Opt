/** Wrapper for a 2D image in cuda stored as a simple linear array */
#include <G3D/G3D.h>
#include <GLG3D/GLG3D.h>
class CudaImage {
protected:
    int m_width;
    int m_height;
    const G3D::ImageFormat* m_format;

    /** On the GPU */
    void* m_data;

    CudaImage(int width, int height, const G3D::ImageFormat* imageFormat, const void* data);

    void updateFromMemory(const void* newData);
    void copyTo(void* dst);

    int sizeInMemory() {
        return m_width*m_height*m_format->cpuBitsPerPixel / 8;
    }

public:

    void* data() {
        return m_data;
    }
    static shared_ptr<CudaImage> createEmpty(int width, int height, const G3D::ImageFormat* imageFormat);
    static shared_ptr<CudaImage> fromTexture(shared_ptr<G3D::Texture> texture);
    void updateTexture(shared_ptr<G3D::Texture> texture);
    void update(shared_ptr<G3D::Texture> texture);

    ~CudaImage();

    
};
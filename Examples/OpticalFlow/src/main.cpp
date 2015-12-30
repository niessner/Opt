#include "main.h"
#include "ImageWarping.h"
#include "ImageHelper.h"


void renderFlowVecotors(ColorImageR8G8B8A8& image, const BaseImage<float2>& flowVectors) {
	const unsigned int skip = 5;	//only every n-th pixel
	
	for (unsigned int j = 0; j < image.getHeight(); j += skip) {
		for (unsigned int i = 0; i < image.getWidth(); i += skip) {
			
			const float2& flowVector = flowVectors(i, j);
			vec2i start = vec2i(i, j);
			vec2i end = start + vec2i(math::round(flowVector.x), math::round(flowVector.y));
			vec4uc color = math::round(255.0f*vec4f(0.1f, 0.8f, 0.1f, 1.0f));	//TODO color-code length

			ImageHelper::drawLine(image, start, end, color);
		}
	}
}

int main(int argc, const char * argv[]) {


	const std::string srcFile = "eval-data/Mequon/frame07.png";
	const std::string tarFile = "eval-data/Mequon/frame09.png";

	ColorImageR8G8B8A8 imageSrc = LodePNG::load(srcFile);
	ColorImageR8G8B8A8 imageTar = LodePNG::load(tarFile);

	ColorImageR32 imageSrcGray = imageSrc.convertToGrayscale();
	ColorImageR32 imageTarGray = imageTar.convertToGrayscale();

	ImageWarping warping(imageSrcGray, imageTarGray);

	BaseImage<float2> flowVectors = warping.solve();

	const std::string outFile = "out.png";
	ColorImageR8G8B8A8 out = imageSrc;
	renderFlowVecotors(out, flowVectors);
	LodePNG::save(out, outFile);

	//const std::string outFile = "out.png";
	//ImageHelper::drawLine(imageSrcGray, vec2ui(50, 50), vec2ui(100, 75), 1.0f);
	//LodePNG::save(ImageHelper::convertGrayToColor(imageSrcGray), outFile);

	
    #ifdef _WIN32
	std::cout << "<press key to continue>" << std::endl;
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

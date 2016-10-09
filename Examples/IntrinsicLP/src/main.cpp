#include "main.h"
#include "ImageWarping.h"

int main(int argc, const char * argv[])
{
	const std::string inputImage = "ye.png";
	
	ColorImageR8G8B8A8	   image = LodePNG::load(inputImage);
	ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x,y) = image(x,y);
		}
	}
	
	ImageWarping warping(imageR32);
	warping.solve();

	ColorImageR32G32B32A32* res = warping.getAlbedo();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp(255.0f*(*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp(255.0f*(*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp(255.0f*(*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "outputAlbedo.png");

	res = warping.getShading();
	ColorImageR8G8B8A8 out2(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(255.0f*math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(255.0f*math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(255.0f*math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out2(x, y) = vec4uc(r, g, b, 255);
		}
	}
	LodePNG::save(out2, "outputShading.png");

	getchar();
	return 0;
}

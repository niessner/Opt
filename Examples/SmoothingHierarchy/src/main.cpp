#include "main.h"
#include "ImageWarping.h"

int main(int argc, const char * argv[])
{
	// MONA
	const std::string inputImage = "MonaSource1_rec.png";

	ColorImageR8G8B8A8	   image = LodePNG::load(inputImage);
	ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x,y) = image(x,y);
		}
	}
	
	ImageWarping warping(imageR32);

	ColorImageR32G32B32A32* res = warping.solve();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "output.png");

	getchar();
	return 0;
}

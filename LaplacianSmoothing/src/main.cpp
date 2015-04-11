
#include "main.h"
#include "Smoothing.h"



int main(int argc, const char * argv[]) {

	const std::string inputImage = "smoothingExampleB.png";
	const ColorImageR8G8B8A8 image = LodePNG::load(inputImage);
	ColorImageR32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x, y) = image(x,y).x;
		}
	}

	Smoothing smoothing(imageR32);

	ColorImageR32 res = smoothing.solve();
	ColorImageR8G8B8A8 out(res.getWidth(), res.getHeight());
	for (unsigned int y = 0; y < res.getHeight(); y++) {
		for (unsigned int x = 0; x < res.getWidth(); x++) {
			unsigned char p = math::round(math::clamp(res(x, y), 0.0f, 255.0f));
			out(x, y) = vec4uc(p, p, p, 255);
		}
	}
	LodePNG::save(out, "tmp.png");

	getchar();

	return 0;
}


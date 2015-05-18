
#include "main.h"
#include "ImageWarping.h"

int main(int argc, const char * argv[]) {


	const std::string inputImage = "smoothingExampleD.png";
	const ColorImageR8G8B8A8 image = LodePNG::load(inputImage);
	ColorImageR32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x, y) = image(x,y).x;
		}
	}

	std::vector<std::vector<int>> constraints; constraints.resize(1);
	constraints[0].push_back(10); constraints[0].push_back(10); constraints[0].push_back(20); constraints[0].push_back(20); // (10, 10) -> (20, 20)

	ImageWarping warping(imageR32, constraints);

	ColorImageR32 res = warping.solve();
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


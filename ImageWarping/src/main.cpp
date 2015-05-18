#include "main.h"
#include "ImageWarping.h"

int main(int argc, const char * argv[]) {

	const std::string inputImage = "smoothingExampleD.png";
	const ColorImageR8G8B8A8 image = LodePNG::load(inputImage);
	ColorImageR32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(y,x) = image(y,x).x;
		}
	}

	std::vector<std::vector<int>> constraints; constraints.resize(2);
	constraints[0].push_back(128); constraints[0].push_back(128); constraints[0].push_back(20); constraints[0].push_back(20);
	constraints[1].push_back(128); constraints[1].push_back(200); constraints[1].push_back(50); constraints[1].push_back(50);

	for (unsigned int i = 0; i < image.getHeight(); i++)
	{
		for (unsigned int j = 0; j < image.getWidth(); j++)
		{
			if (i == 0 || j == 0 || i == (image.getHeight() - 1) || j == (image.getWidth() - 1))
			{
				std::vector<int> v; v.push_back(i); v.push_back(j); v.push_back(i); v.push_back(j);
				constraints.push_back(v);
			}
		}
	}

	ImageWarping warping(imageR32, constraints);

	ColorImageR32 res = warping.solve();
	ColorImageR8G8B8A8 out(res.getWidth(), res.getHeight());
	for (unsigned int y = 0; y < res.getHeight(); y++) {
		for (unsigned int x = 0; x < res.getWidth(); x++) {
			unsigned char p = math::round(math::clamp(res(y, x), 0.0f, 255.0f));
			out(y, x) = vec4uc(p, p, p, 255);
		}
	}
	LodePNG::save(out, "output.png");

	getchar();

	return 0;
}

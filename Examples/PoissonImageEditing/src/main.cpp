#include "main.h"
#include "ImageWarping.h"

int main(int argc, const char * argv[]) {

	// JRK
	const std::string inputImage	 = "jrkSource2.png";
	const std::string inputImage1	 = "jrkSource1.png";
	const std::string inputImageMask = "jrkMask.png";
	const unsigned int offsetX = 0;
	const unsigned int offsetY = 0;
	const bool invertMask = true;

	//// MONA
	//const std::string inputImage = "MonaSource1.png";
	//const std::string inputImage1 = "MonaSource2.png";
	//const std::string inputImageMask = "MonaMask.png";
	//const unsigned int offsetX = 37;
	//const unsigned int offsetY = 88;
	//const bool invertMasek = false;

	ColorImageR8G8B8A8	   image = LodePNG::load(inputImage);
	ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x,y) = image(x,y);
		}
	}

	ColorImageR8G8B8A8	   image1 = LodePNG::load(inputImage1);
	ColorImageR32G32B32A32 imageR321(image1.getWidth(), image1.getHeight());
	for (unsigned int y = 0; y < image1.getHeight(); y++) {
		for (unsigned int x = 0; x < image1.getWidth(); x++) {
			imageR321(x, y) = image1(x, y);
		}
	}

	ColorImageR32G32B32A32 image1Large = imageR32;
	image1Large.setPixels(ml::vec4uc(0, 0, 0, 255));
	for (unsigned int y = 0; y < imageR321.getHeight(); y++) {
		for (unsigned int x = 0; x < imageR321.getWidth(); x++) {
			image1Large(x + offsetY, y + offsetX) = imageR321(x, y);
		}
	}

	//ColorImageR8G8B8A8 tmp(image1Large.getWidth(), image1Large.getHeight());
	//for (unsigned int y = 0; y < image1Large.getHeight(); y++) {
	//	for (unsigned int x = 0; x < image1Large.getWidth(); x++) {
	//		tmp(x, y) = math::round(image1Large(x, y) * 255);
	//	}
	//}
	//LodePNG::save(tmp, "tmp.png");

	
	const ColorImageR8G8B8A8 imageMask = LodePNG::load(inputImageMask);
	ColorImageR32 imageR32Mask(imageMask.getWidth(), imageMask.getHeight());
	for (unsigned int y = 0; y < imageMask.getHeight(); y++) {
		for (unsigned int x = 0; x < imageMask.getWidth(); x++) {
			unsigned char c = imageMask(x, y).x;
			if (invertMask) {
				if (c == 255) c = 0;
				else c = 255;
			}

			imageR32Mask(x, y) = c;
		}
	}

	ColorImageR32 imageR32MaskLarge(image.getWidth(), image.getHeight());
	imageR32MaskLarge.setPixels(0);
	for (unsigned int y = 0; y < imageMask.getHeight(); y++) {
		for (unsigned int x = 0; x < imageMask.getWidth(); x++) {
			imageR32MaskLarge(x + offsetY, y + offsetX) = imageR32Mask(x, y);
		}
	}
	
	ImageWarping warping(imageR32, image1Large, imageR32MaskLarge);

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

#include "main.h"
#include "ImageWarping.h"

static void loadConstraints(std::vector<std::vector<int> >& constraints, std::string filename) {
  std::ifstream in(filename, std::fstream::in);

	if(!in.good())
	{
		std::cout << "Could not open marker file " << filename << std::endl;
		assert(false);
	}

	unsigned int nMarkers;
	in >> nMarkers;
	constraints.resize(nMarkers);
	for(unsigned int m = 0; m<nMarkers; m++)
	{
		int temp;
		for (int i = 0; i < 4; ++i) {
			in >> temp;
			constraints[m].push_back(temp);
		}

	}

	in.close();
}


int main(int argc, const char * argv[]) {

	//// DOG
	//const std::string inputImage = "smoothingExampleD.png";
	//const std::string inputImageMask = "smoothingExampleDMask.png";
	//std::vector<std::vector<int>> constraints; constraints.resize(4);
	//constraints[0].push_back(95);  constraints[0].push_back(118); constraints[0].push_back(80);  constraints[0].push_back(118);
	//constraints[1].push_back(120); constraints[1].push_back(118); constraints[1].push_back(130); constraints[1].push_back(118);
	//constraints[2].push_back(163); constraints[2].push_back(111); constraints[2].push_back(153); constraints[2].push_back(111);
	//constraints[3].push_back(183); constraints[3].push_back(111); constraints[3].push_back(193); constraints[3].push_back(111);


	// PILLAR
	/*
	const std::string inputImage = "bend2.png";
	const std::string inputImageMask = "bendMask.png";
	std::vector<std::vector<int>> constraints; constraints.resize(3);
	constraints[0].push_back(48); constraints[0].push_back(61); constraints[0].push_back(144); constraints[0].push_back(69);
	constraints[1].push_back(64); constraints[1].push_back(61); constraints[1].push_back(154); constraints[1].push_back(82);
	constraints[2].push_back(80); constraints[2].push_back(61); constraints[2].push_back(165); constraints[2].push_back(92);
	*/


	// CAT
	const std::string inputImage = "cartooncat.png";
	const std::string inputImageMask = "catmask.png";
	std::vector<std::vector<int>> constraints;
	loadConstraints(constraints, "cat.constraints");
	

	
	ColorImageR8G8B8A8 image = LodePNG::load(inputImage);
	ColorImageR32 imageR32(image.getWidth(), image.getHeight());
	//printf("width %d, height %d\n", image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			//printf("x %d, y %d\n", x, y);

			imageR32(x,y) = image(x,y).x;
		}
	}

	const ColorImageR8G8B8A8 imageMask = LodePNG::load(inputImageMask);
	ColorImageR32 imageR32Mask(imageMask.getWidth(), imageMask.getHeight());
	for (unsigned int y = 0; y < imageMask.getHeight(); y++) {
		for (unsigned int x = 0; x < imageMask.getWidth(); x++) {
			imageR32Mask(x, y) = imageMask(x, y).x;
		}
	}
	
	
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

	ImageWarping warping(imageR32, imageR32Mask, constraints);

	ColorImageR32* res = warping.solve();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char p = math::round(math::clamp((*res)(x, y), 0.0f, 255.0f));
			out(x, y) = vec4uc(p, p, p, 255);
	
			for (unsigned int k = 0; k < constraints.size(); k++)
			{
				if (constraints[k][2] == y && constraints[k][3] == x) 
				{
					if (imageR32Mask(constraints[k][0], constraints[k][1]) == 0)
					{
						out(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
		
				if (constraints[k][0] == y && constraints[k][1] == x)
				{
					if (imageR32Mask(x, y) == 0)
					{
						image(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
			}
		}
	}
	LodePNG::save(out, "output.png");
	LodePNG::save(image, "inputMark.png");

    #ifdef _WIN32
 	    getchar();
    #else
        exit(0);
    #endif

	return 0;
}

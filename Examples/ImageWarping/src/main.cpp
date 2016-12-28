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

	std::string inputImage = "cartooncat.png";
	std::string inputImageMask = "catmask.png";
	std::vector<std::vector<int>> constraints;
	loadConstraints(constraints, "cat.constraints");
    int downsampleFactor = 1;

	bool lmOnlyFullSolve = false;
    if (argc > 1) {
        downsampleFactor = atoi(argv[1]);
        if (downsampleFactor > 0) {
            inputImage = "cartooncat512.png";
            inputImageMask = "catmask512_black.png";
            constraints.clear();
            loadConstraints(constraints, "cat512.constraints");
			//lmOnlyFullSolve = true;
        } else {
            downsampleFactor = 1;
        }
        
    }
    bool performanceRun = false;
    if (argc > 2) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
            if (atoi(argv[1]) > 0) {
                lmOnlyFullSolve = true;
            }
        } else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }

    


	ColorImageR8G8B8A8 image = LodePNG::load(inputImage);
    const ColorImageR8G8B8A8 imageMask = LodePNG::load(inputImageMask);


    
    ColorImageR32G32B32 imageColor(image.getWidth() / downsampleFactor, image.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < image.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < image.getWidth() / downsampleFactor; x++) {
            auto val = image(x*downsampleFactor, y*downsampleFactor);

            imageColor(x,y) = vec3f(val.x, val.y, val.z);
        }
    }

    ColorImageR32 imageR32(imageColor.getWidth(), imageColor.getHeight());
    printf("width %d, height %d\n", imageColor.getWidth(), imageColor.getHeight());
    for (unsigned int y = 0; y < imageColor.getHeight(); y++) {
        for (unsigned int x = 0; x < imageColor.getWidth(); x++) {
            imageR32(x, y) = imageColor(x, y).x;
		}
	}
    int activePixels = 0;

    ColorImageR32 imageR32Mask(imageMask.getWidth() / downsampleFactor, imageMask.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < imageMask.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < imageMask.getWidth() / downsampleFactor; x++) {
            imageR32Mask(x, y) = imageMask(x*downsampleFactor, y*downsampleFactor).x;
            if (imageMask(x*downsampleFactor, y*downsampleFactor).x == 0.0f) {
                ++activePixels;
            }
		}
	}
    printf("numActivePixels: %d\n", activePixels);
	
    for (auto& constraint : constraints) {
        for (auto& c : constraint) {
            c /= downsampleFactor;
        }
    }

    for (unsigned int y = 0; y < imageColor.getHeight(); y++)
	{
        for (unsigned int x = 0; x < imageColor.getWidth(); x++)
		{
            if (y == 0 || x == 0 || y == (imageColor.getHeight() - 1) || x == (imageColor.getWidth() - 1))
			{
				std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
				constraints.push_back(v);
			}
		}
	}

	ImageWarping warping(imageR32, imageColor, imageR32Mask, constraints, performanceRun, lmOnlyFullSolve);

	ColorImageR32G32B32* res = warping.solve();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = util::boundToByte((*res)(x, y).x);
            unsigned char g = util::boundToByte((*res)(x, y).y);
            unsigned char b = util::boundToByte((*res)(x, y).z);
			out(x, y) = vec4uc(r, g, b, 255);
	
			for (unsigned int k = 0; k < constraints.size(); k++)
			{
				if (constraints[k][2] == x && constraints[k][3] == y) 
				{
                    if (imageR32Mask(constraints[k][0], constraints[k][1]) == 0)
					{
						//out(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
		
				if (constraints[k][0] == x && constraints[k][1] == y)
				{
					if (imageR32Mask(x, y) == 0)
					{
                        image(x*downsampleFactor, y*downsampleFactor) = vec4uc(255, 0, 0, 255);
					}
				}
			}
		}
	}
	LodePNG::save(out, "output.png");
	LodePNG::save(image, "inputMark.png");
    printf("Saved\n");

	return 0;
}

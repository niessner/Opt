
#include "main.h"

struct App
{
    void go();

    BundlerManager bundler;
};

void App::go()
{
    //bundler.loadSensorFileA(constants::dataDir + "/sensors/sample.sensor");
    bundler.loadSensorFileB(constants::dataDir + "/sensors/fr3_office.sens", 10);
    //bundler.frames.resize(2);
    //bundler.loadSensorFileB(constants::dataDir + "/sensors/fr1_desk.sens", 10);
    bundler.computeKeypoints();
    bundler.addAllCorrespondences(16);

    util::makeDirectory(constants::debugDir);

    bundler.allCorrespondences[0].visualize(constants::debugDir + "0_1/");

    ofstream alignmentFile(constants::debugDir + "alignmentError.csv");
    
    //const vector<double> thresholds = { -1.0, 0.1, 0.05, 0.02, 0.01, 0.005 };
    const vector<double> thresholds = { -1.0 };

    for (auto &threshold : iterate(thresholds))
    {
        if (threshold.value > 0.0)
            bundler.thresholdCorrespondences(threshold.value);

        //bundler.solveOpt();

        /*for (auto &f : bundler.frames)
        {
            for (int i = 0; i < 6; i++)
            {
                //cout << f.camera[i] << " ";
                //f.camera[i] = 0.0;
            }
            //cout << endl;
        }*/

        bundler.solveCeres(1e-20);
        

        const string suffix = util::zeroPad((int)threshold.index, 1);
        bundler.saveKeypointCloud(constants::debugDir + "result" + suffix + ".ply");
        bundler.visualizeCameras(constants::debugDir + "result" + suffix + "Cameras.ply");
        bundler.saveResidualDistribution(constants::debugDir + "residuals" + suffix + ".csv");
        alignmentFile << suffix << "," << threshold.value << "," << bundler.globalError() << endl;
    }
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    App app;
    app.go();

    return 0;
}

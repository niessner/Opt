
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
    //bundler.loadSensorFileB(constants::dataDir + "/sensors/fr1_desk.sens", 10);
    bundler.computeKeypoints();
    bundler.addAllCorrespondences(80);
    /*bundler.addCorrespondences(1);
    bundler.addCorrespondences(2);
    bundler.addCorrespondences(3);
    bundler.addCorrespondences(4);
    bundler.addCorrespondences(10);
    bundler.addCorrespondences(20);
    bundler.addCorrespondences(40);
    bundler.addCorrespondences(80);
    bundler.addCorrespondences(120);
    bundler.addCorrespondences(160);*/

    bundler.allCorrespondences[0].visualize(constants::debugDir + "0_1/");

    ofstream alignmentFile(constants::debugDir + "alignmentError.csv");
        
    bundler.solve(1e-20);
    util::makeDirectory(constants::debugDir);
    bundler.saveKeypointCloud(constants::debugDir + "resultA.ply");
    bundler.visualizeCameras(constants::debugDir + "resultACameras.ply");
    bundler.saveResidualDistribution(constants::debugDir + "residualsA.csv");
    alignmentFile << bundler.globalError() << endl;
    
    bundler.thresholdCorrespondences(0.1);
    bundler.solve(1e-20);
    bundler.saveKeypointCloud(constants::debugDir + "resultB.ply");
    bundler.visualizeCameras(constants::debugDir + "resultBCameras.ply");
    bundler.saveResidualDistribution(constants::debugDir + "residualsB.csv");
    alignmentFile << bundler.globalError() << endl;

    bundler.thresholdCorrespondences(0.05);
    bundler.solve(1e-20);
    bundler.saveKeypointCloud(constants::debugDir + "resultC.ply");
    bundler.visualizeCameras(constants::debugDir + "resultCCameras.ply");
    bundler.saveResidualDistribution(constants::debugDir + "residualsC.csv");
    alignmentFile << bundler.globalError() << endl;

    bundler.thresholdCorrespondences(0.02);
    bundler.solve(1e-20);
    bundler.saveKeypointCloud(constants::debugDir + "resultD.ply");
    bundler.visualizeCameras(constants::debugDir + "resultDCameras.ply");
    bundler.saveResidualDistribution(constants::debugDir + "residualsD.csv");
    alignmentFile << bundler.globalError() << endl;

    bundler.thresholdCorrespondences(0.01);
    bundler.solve(1e-20);
    bundler.saveKeypointCloud(constants::debugDir + "resultE.ply");
    bundler.visualizeCameras(constants::debugDir + "resultECameras.ply");
    bundler.saveResidualDistribution(constants::debugDir + "residualsE.csv");
    alignmentFile << bundler.globalError() << endl;
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    App app;
    app.go();

    return 0;
}

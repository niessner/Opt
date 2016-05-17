
namespace constants
{
    const double SIFTHessian = 500.0;

    const int minCorrespondenceCount = 20;
    const int minInlierCount = 10;

    const int RANSACSamples = 3;

    const int RANSACEarlyIters = 50;
    const int RANSACEarlyInlierMin = 100;

    const int RANSACFullIters = 1000;
    
    const float outlierDist = 0.02f;
    const float outlierDistSq = outlierDist * outlierDist;

    const double CERESTolerance = 1e-8;

    const string dataDir = R"(..\data\)";
    const string debugDir = dataDir + "debug/";

    //const int debugMaxFrameCount = 20;
    const int debugMaxFrameCount = numeric_limits<int>::max();

}
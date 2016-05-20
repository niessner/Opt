
struct BundlerManager
{
    void loadSensorFileA(const string &filename);
    void loadSensorFileB(const string &filename, int frameSkip);

    vec2i imagePixelToDepthPixel(const vec2f &imageCoord) const;

    void computeKeypoints();

    void addAllCorrespondences(int maxSkip);
    void computeCorrespondences(int forwardSkip, vector<ImagePairCorrespondences> &result);
    void addCorrespondences(int imageAIndex, int imageBIndex, vector<ImagePairCorrespondences> &result);

    void visualize(const string &dir, int imageAIndex, int imageBIndex) const;

    void solveCeres(double tolerance);
    void solveOpt();
    void alignToGroundTruth();
    void updateResiduals();
    void thresholdCorrespondences(double cutoff);

    void saveResidualDistribution(const string &filename) const;

    void saveKeypointCloud(const string &outputFilename) const;

    void visualizeCameras(const string &filename) const;
    double globalError() const;

    vector<BundlerFrame> frames;
    vector<ImagePairCorrespondences> allCorrespondences;
};

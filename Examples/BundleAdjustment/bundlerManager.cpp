
#include "main.h"

extern "C" {
#include "Opt.h"
}

#define cutilSafeCall(x) (x);

vec2i BundlerManager::imagePixelToDepthPixel(const vec2f &imageCoord) const
{
    // assumes remapped sensor file!
    return math::round(imageCoord);
}

void BundlerManager::loadSensorFileA(const string &filename)
{
    cout << "Reading sensor file (old format)" << endl;
    BinaryDataStreamFile in(filename, false);
    CalibratedSensorData data;
    in >> data;
    in.closeStream();

    if (data.m_ColorImageWidth != data.m_DepthImageWidth ||
        data.m_ColorImageHeight != data.m_DepthImageHeight ||
        data.m_ColorImages.size() != data.m_DepthImages.size())
    {
        cout << "Sensor file not remapped" << endl;
        return;
    }

    const int width = data.m_ColorImageWidth;
    const int height = data.m_ColorImageHeight;
    const int frameCount = min((int)data.m_ColorImages.size(), constants::debugMaxFrameCount);

    cout << "Creating frames" << endl;
    frames.resize(frameCount);

    for (const auto &frame : iterate(frames))
    {
        frame.value.index = (int)frame.index;
        frame.value.colorImage.allocate(width, height);
        frame.value.depthImage.allocate(width, height);
        frame.value.depthIntrinsicInverse = data.m_CalibrationDepth.m_IntrinsicInverse;

        memcpy(frame.value.colorImage.getData(), data.m_ColorImages[frame.index], sizeof(vec4uc) * width * height);
        memcpy(frame.value.depthImage.getData(), data.m_DepthImages[frame.index], sizeof(float) * width * height);

        SAFE_DELETE_ARRAY(data.m_ColorImages[frame.index]);
        SAFE_DELETE_ARRAY(data.m_DepthImages[frame.index]);
    }
}

void BundlerManager::loadSensorFileB(const string &filename, unsigned int frameSkip, unsigned int maxNumFrames)
{
    cout << "Reading sensor file (new format)" << endl;
    SensorData data(filename);
    if (data.m_colorWidth != data.m_depthWidth ||
        data.m_colorHeight != data.m_depthHeight)
    {
        cout << "Sensor file not remapped" << endl;
        return;
    }

    const unsigned int width = data.m_colorWidth;
    const unsigned int height = data.m_colorHeight;
    const unsigned int pixelCount = width * height;
    const unsigned int baseFrameCount = (int)data.m_frames.size();
    unsigned int newFrameCount = baseFrameCount / frameSkip;

    cout << "Creating frames" << endl;
	if (maxNumFrames != 0) newFrameCount = std::min(maxNumFrames, newFrameCount);
    frames.resize(newFrameCount);
    
    int baseFrameIndex = 0;
    for (const auto &frame : iterate(frames))
    {
        frame.value.index = (int)frame.index;
        frame.value.colorImage.allocate(width, height);
        frame.value.depthImage.allocate(width, height);
        frame.value.depthIntrinsicInverse = data.m_calibrationDepth.m_intrinsic.getInverse();

        auto &sensorFrame = data.m_frames[baseFrameIndex];
        
        frame.value.groundTruthCamera = Cameraf(sensorFrame.getCameraToWorld(), 60.0f, 1.0f, 0.01f, 100.0f);        

        vec3uc* colorDataVec3uc = data.decompressColorAlloc(baseFrameIndex);        
        unsigned short* depthDataUShort = data.decompressDepthAlloc(baseFrameIndex); 

        for (unsigned int i = 0; i < pixelCount; i++)
        {
			frame.value.depthImage.getData()[i] = (float)depthDataUShort[i] / data.m_depthShift;
			frame.value.colorImage.getData()[i] = vec4uc(colorDataVec3uc[i], 255);
        }        

        std::free(colorDataVec3uc);
        std::free(depthDataUShort);

        baseFrameIndex += frameSkip;
    }
}

void BundlerManager::computeKeypoints()
{
    cout << "Computing image keypoints for " << frames.size() << " frames" << endl;
    FeatureExtractor extractor;
    for (auto &i : frames)
    {
        i.keypoints = extractor.detectAndDescribe(i.colorImage);
    }
}

void BundlerManager::addAllCorrespondences(int maxSkip)
{
    maxSkip = min(maxSkip, (int)frames.size());
#pragma omp parallel for schedule(dynamic,1) num_threads(8)
    for (int skip = 1; skip < maxSkip; skip++)
    {
        vector<ImagePairCorrespondences> result;
        computeCorrespondences(skip, result);
#pragma omp critical
        {
            for (auto &c : result)
                allCorrespondences.push_back(c);
        }
    }
}

void BundlerManager::computeCorrespondences(int forwardSkip, vector<ImagePairCorrespondences> &result)
{
    cout << "Adding correspondences (skip=" << forwardSkip << ")" << endl;
    for (auto &startImage : frames)
    {
        addCorrespondences(startImage.index, startImage.index + forwardSkip, result);
    }
}

void BundlerManager::addCorrespondences(int frameAIndex, int frameBIndex, vector<ImagePairCorrespondences> &result)
{
    if (frameBIndex >= frames.size())
        return;

    const BundlerFrame &imageA = frames[frameAIndex];
    const BundlerFrame &imageB = frames[frameBIndex];

    KeypointMatcher matcher;
    auto matches = matcher.match(imageA.keypoints, imageB.keypoints);

    ImagePairCorrespondences correspondences;
    correspondences.imageA = &frames[frameAIndex];
    correspondences.imageB = &frames[frameBIndex];

    for (auto &match : matches)
    {
        ImageCorrespondence corr;
        
        corr.imageA = frameAIndex;
        corr.imageB = frameBIndex;

        corr.keyPtDist = match.distance;

        const vec2i depthPixelA = imagePixelToDepthPixel(imageA.keypoints[match.indexA].pt);
        const vec2i depthPixelB = imagePixelToDepthPixel(imageB.keypoints[match.indexB].pt);

        corr.ptAPixel = math::round(imageA.keypoints[match.indexA].pt);
        corr.ptALocal = imageA.localPos(depthPixelA);

        corr.ptBPixel = math::round(imageB.keypoints[match.indexB].pt);
        corr.ptBLocal = imageB.localPos(depthPixelB);

        corr.residual = -1.0f;

        if (corr.ptALocal.isValid() &&
            corr.ptBLocal.isValid())
            correspondences.allCorr.push_back(corr);
    }

    correspondences.estimateTransform();

    if (correspondences.transformInliers >= constants::minInlierCount)
        result.push_back(correspondences);
}

void BundlerManager::solveCeres(double tolerance)
{
    ceres::Problem problem;
    int totalCorrespondences = 0;
    for (const ImagePairCorrespondences &iCorr : allCorrespondences)
    {
        BundlerFrame &frameA = *iCorr.imageA;
        BundlerFrame &frameB = *iCorr.imageB;

        for (const ImageCorrespondence &c : iCorr.inlierCorr)
        {
            ceres::CostFunction* costFunction = CorrespondenceCostFunc::Create(c);
            problem.AddResidualBlock(costFunction, NULL, frameA.camera, frameB.camera);
            totalCorrespondences++;
        }
    }

    for (const Keypoint &keypt : frames[0].keypoints)
    {
        const vec3f framePos = frames[0].localPos(keypt.pt);
        if (!framePos.isValid())
            continue;

        ceres::CostFunction* costFunction = AnchorCostFunc::Create(framePos, 100.0f);
        problem.AddResidualBlock(costFunction, NULL, frames[0].camera);
    }

    cout << "Total correspondences: " << totalCorrespondences << endl;
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100000;
    options.max_num_consecutive_invalid_steps = 100;
    options.num_threads = 7;
    options.num_linear_solver_threads = 7;
    options.function_tolerance = tolerance;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    for (BundlerFrame &f : frames)
        f.updateTransforms(mat4f::identity());

    alignToGroundTruth();
    updateResiduals();
}

template <class type> type* createDeviceBuffer(const vector<type>& v, const string &bufferName) {

    const bool debugBuffers = true;
    if (debugBuffers)
    {
        static int debugCounter = 0;
        ofstream file(bufferName + ".txt");
        for (auto &e : v)
            file << e << endl;
    }

    type* d_ptr;
    cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));
    cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
    return d_ptr;
}

void BundlerManager::solveOpt(int linearIterations, int nonLinearIterations)
{
    Opt_State*		optimizerState;
    Opt_Problem*	problem;
    Opt_Plan*		plan;

    int correspondenceCount = 0;
    for (const ImagePairCorrespondences &iCorr : allCorrespondences)
        for (const ImageCorrespondence &c : iCorr.inlierCorr)
            correspondenceCount++;

    cout << "Total correspondences: " << correspondenceCount << endl;

    const int cameraCount = (int)frames.size();
    unsigned int dims[] = { (unsigned int)cameraCount,  (unsigned int) correspondenceCount };

    optimizerState = Opt_NewState();
    //LMGPU, gaussNewtonGPU
    problem = Opt_ProblemDefine(optimizerState, "bundleAdjustment.t", "gaussNewtonGPU");
    plan = Opt_ProblemPlan(optimizerState, problem, dims);

    unsigned int blockIterations = 1;	//not used

    void* solverParams[] = { &nonLinearIterations, &linearIterations, &blockIterations };

    vector<float> cameras(cameraCount * 6);
    for (BundlerFrame &f : frames)
    {
        for (int i = 0; i < 6; i++)
            cameras[f.index * 6 + i] = (float)f.camera[i];
    }
    float *d_cameras = createDeviceBuffer(cameras, "cameras");
    

    vector<vec3f> correspondences;
    vector<int> cameraAIndices, cameraBIndices, correspondenceIndices;
    for (const ImagePairCorrespondences &iCorr : allCorrespondences)
    {
        for (const ImageCorrespondence &c : iCorr.inlierCorr)
        {
            correspondences.push_back(c.ptALocal);
            correspondences.push_back(c.ptBLocal);
            cameraAIndices.push_back(c.imageA);
            cameraBIndices.push_back(c.imageB);
            correspondenceIndices.push_back((int)correspondenceIndices.size());
        }
    }
    vec3f *d_correspondences = createDeviceBuffer(correspondences, "correspondences");

    vector<float> anchorWeights(cameraCount, 0.0f);
    anchorWeights[0] = 100.0f;
    float *d_anchorWeights = createDeviceBuffer(anchorWeights, "anchorWeights");
    int *d_cameraAIndices = createDeviceBuffer(cameraAIndices, "cameraAIndices");
    int *d_cameraBIndices = createDeviceBuffer(cameraBIndices, "cameraBIndices");
    int *d_correspondenceIndices = createDeviceBuffer(correspondenceIndices, "correspondenceIndices");

    void* problemParams[] = { d_cameras, d_correspondences, d_anchorWeights, &correspondenceCount, d_cameraAIndices, d_cameraBIndices, d_correspondenceIndices };
    
    Opt_ProblemSolve(optimizerState, plan, problemParams, solverParams);

    cutilSafeCall(cudaMemcpy(cameras.data(), d_cameras, sizeof(float)*cameras.size(), cudaMemcpyDeviceToHost));

    for (BundlerFrame &f : frames)
    {
        for (int i = 0; i < 6; i++)
            f.camera[i] = cameras[f.index * 6 + i];
        f.updateTransforms(mat4f::identity());
    }

    alignToGroundTruth();
    updateResiduals();
}

void BundlerManager::alignToGroundTruth()
{
    vector<vec3f> source, target;
    vec3f eigenvalues;

    for (BundlerFrame &f : frames)
    {
        source.push_back(f.debugCamera.getEye());
        target.push_back(f.groundTruthCamera.getEye());
    }

    mat4f globalAlignmentTransform = mat4f::identity();
    if (frames.size() >= 3)
        globalAlignmentTransform = EigenWrapperf::kabsch(source, target, eigenvalues);

    for (BundlerFrame &f : frames)
        f.updateTransforms(globalAlignmentTransform);
}

void BundlerManager::saveKeypointCloud(const string &outputFilename) const
{
    PointCloudf cloud;
    
    for (const BundlerFrame &frame : frames)
    {
        const int stride = 5;
        for (const auto &p : frame.depthImage)
        {
            vec2i coord((int)p.x, (int)p.y);
            const vec3f framePos = frame.localPos(coord);
            if (!framePos.isValid())
                continue;

            if (p.x % stride != 0 || p.y % stride != 0)
                continue;

            cloud.m_points.push_back(frame.frameToWorld * framePos);
            cloud.m_colors.push_back(vec4f(frame.colorImage(coord)) / 255.0f);
        }
    }

    PointCloudIOf::saveToFile(outputFilename, cloud);
}

void BundlerManager::updateResiduals()
{
    for (ImagePairCorrespondences &iCorr : allCorrespondences)
    {
        const BundlerFrame &frameA = *iCorr.imageA;
        const BundlerFrame &frameB = *iCorr.imageB;

        for (ImageCorrespondence &c : iCorr.inlierCorr)
        {
            const vec3f ptAWorld = frameA.frameToWorld * c.ptALocal;
            const vec3f ptBWorld = frameB.frameToWorld * c.ptBLocal;
            c.residual = vec3f::dist(ptAWorld, ptBWorld);
        }

        sort(iCorr.inlierCorr.begin(), iCorr.inlierCorr.end());
    }
}

void BundlerManager::thresholdCorrespondences(double cutoff)
{
    for (ImagePairCorrespondences &iCorr : allCorrespondences)
    {
        vector<ImageCorrespondence> newCorr;
        for (ImageCorrespondence &c : iCorr.inlierCorr)
        {
            if (c.residual < cutoff)
                newCorr.push_back(c);
        }
        iCorr.inlierCorr = newCorr;
    }
}

void BundlerManager::saveResidualDistribution(const string &filename) const
{
    ofstream file(filename);
    file << "i0,i1,inliers";

    const int quartiles = 10;
    for (int i = 0; i < quartiles; i++)
        file << ",q" << i;
    file << endl;

    for (const ImagePairCorrespondences &iCorr : allCorrespondences)
    {
        file << iCorr.imageA->index << ",";
        file << iCorr.imageB->index << ",";
        file << iCorr.inlierCorr.size();

        for (int i = 0; i < quartiles; i++)
        {
            const double s = (double)i / (quartiles - 1.0);
            int index = math::clamp(math::round(s * iCorr.inlierCorr.size()), 0, (int)iCorr.inlierCorr.size() - 1);
            file << "," << iCorr.inlierCorr[index].residual;
        }
        file << endl;
    }
}

double BundlerManager::globalError() const
{
    double sum = 0.0;
    for (const BundlerFrame &f : frames)
    {
        const vec3f p0 = f.debugCamera.getEye();
        const vec3f p1 = f.groundTruthCamera.getEye();
        sum += vec3f::dist(p0, p1);
    }
    return sum / (double)frames.size();
}

void BundlerManager::visualizeCameras(const string &filename) const
{
    vector<TriMeshf> meshes;

    const vec4f AColor(1.0f, 0.5f, 0.5f, 1.0f);
    const vec4f BColor(0.5f, 0.5f, 1.0f, 1.0f);

    auto makeColoredBox = [](const vec3f &center, const vec4f &color, float radius) {
        TriMeshf result = ml::Shapesf::box(radius, radius, radius);
        result.transform(mat4f::translation(center));
        result.setColor(color);
        return result;
    };

    for (auto &frame : frames)
    {
        meshes.push_back(makeColoredBox(frame.debugCamera.getEye(), AColor, 0.02f));
        meshes.push_back(makeColoredBox(frame.groundTruthCamera.getEye(), BColor, 0.02f));
    }

    TriMeshf unifiedMesh = Shapesf::unifyMeshes(meshes);

    MeshIOf::saveToPLY(filename, unifiedMesh.getMeshData());
}


#include "main.h"

vec3f BundlerFrame::localPos(const vec2i &depthPixel) const
{
    if (!depthImage.isValidCoordinate(depthPixel))
        return vec3f(numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), numeric_limits<float>::infinity());

    const float depth = depthImage(depthPixel);
    if (!depthImage.isValidValue(depth) || depth <= 1e-6 || depth >= constants::maxDepth)
        return vec3f(numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), numeric_limits<float>::infinity());

    const vec4f world = depthIntrinsicInverse * vec4f((float)depthPixel.x * depth, (float)depthPixel.y * depth, depth, 0.0f);
    return world.getVec3();
}

void ImagePairCorrespondences::visualize(const string &dir) const
{
    util::makeDirectory(dir);

    Bitmap imageAVizAll = imageA->colorImage, imageBVizAll = imageB->colorImage;
    for (auto &c : allCorr)
    {
        const vec4uc matchColor = helper::randomMatchColor();
        helper::splatPoint(imageAVizAll, c.ptAPixel, matchColor);
        helper::splatPoint(imageBVizAll, c.ptBPixel, matchColor);
    }

    Bitmap imageAVizInlier = imageA->colorImage, imageBVizInlier = imageB->colorImage;
    for (auto &c : inlierCorr)
    {
        const vec4uc matchColor = helper::randomMatchColor();
        helper::splatPoint(imageAVizInlier, c.ptAPixel, matchColor);
        helper::splatPoint(imageBVizInlier, c.ptBPixel, matchColor);
    }

    LodePNG::save(imageAVizAll, dir + "A_all.png");
    LodePNG::save(imageBVizAll, dir + "B_all.png");
    LodePNG::save(imageAVizInlier, dir + "A_inl.png");
    LodePNG::save(imageBVizInlier, dir + "B_inl.png");

    PointCloudf cloud;

    vector<TriMeshf> meshes;

    const vec3f AOffset = vec3f::eX * 0.1f;
    const vec3f BOffset = vec3f::eX * -0.1f;

    const vec4f AColor(1.0f, 0.5f, 0.5f, 1.0f);
    const vec4f BColor(0.5f, 0.5f, 1.0f, 1.0f);

    auto makeColoredBox = [](const vec3f &center, const vec4f &color, float radius) {
        TriMeshf result = ml::Shapesf::box(radius, radius, radius);
        result.transform(mat4f::translation(center));
        result.setColor(color);
        return result;
    };

    auto addMatch = [&](const ImageCorrespondence &m, const vec4f &matchColor, float scale)
    {
        cloud.m_points.push_back(m.ptALocal + AOffset);
        cloud.m_colors.push_back(matchColor);

        cloud.m_points.push_back(m.ptBLocal + BOffset);
        cloud.m_colors.push_back(matchColor);

        meshes.push_back(makeColoredBox(m.ptALocal + AOffset, matchColor, 0.015f * scale));
        meshes.push_back(makeColoredBox(m.ptBLocal + BOffset, matchColor, 0.015f * scale));

        const TriMeshf cylinder = Shapesf::cylinder(m.ptALocal + AOffset, m.ptBLocal + BOffset, 0.007f * scale, 2, 4, matchColor);
        meshes.push_back(cylinder);
    };

    for (auto &m : inlierCorr)
    {
        vec4f matchColor(util::randomUniform(0.3f, 1.0f),
            util::randomUniform(0.3f, 1.0f),
            util::randomUniform(0.3f, 1.0f), 1.0f);
        addMatch(m, matchColor, 1.0f);
    }

    for (auto &m : allCorr)
    {
        vec4f matchColor(0.5f, 0.5f, 0.5f, 1.0f);
        addMatch(m, matchColor, 0.6f);
    }

    const int stride = 2;
    for (const auto &p : imageA->depthImage)
    {
        const vec3f pos = imageA->localPos(vec2i((int)p.x, (int)p.y));
        if (!pos.isFinite())
            continue;

        if (p.x % stride != 0 || p.y % stride != 0)
            continue;

        meshes.push_back(makeColoredBox(pos + AOffset, AColor, 0.002f));
    }

    for (const auto &p : imageB->depthImage)
    {
        const vec3f pos = imageB->localPos(vec2i((int)p.x, (int)p.y));
        if (!pos.isFinite())
            continue;

        if (p.x % stride != 0 || p.y % stride != 0)
            continue;

        meshes.push_back(makeColoredBox(pos + BOffset, BColor, 0.002f));
    }

    TriMeshf unifiedMesh = Shapesf::unifyMeshes(meshes);

    MeshIOf::saveToPLY(dir + "mesh.ply", unifiedMesh.makeMeshData());
}

void ImagePairCorrespondences::estimateTransform()
{
    if (allCorr.size() < constants::minCorrespondenceCount)
    {
        transformInliers = 0;
        //cout << "Too few correspondences: " << allCorr.size() << endl;
        return;
    }

    mat4f bestTransform = mat4f::identity();
    TransformResult bestStats;
    bestStats.totalError = std::numeric_limits<double>::max();
    for (int ransacIter = 0; ransacIter < constants::RANSACFullIters; ransacIter++)
    {
        set<int> indices;
        while (indices.size() < constants::RANSACSamples)
        {
            const int index = util::randomInteger(0, (int)allCorr.size() - 1);
            indices.insert(index);
        }
        
        const mat4f candidateTransform = estimateTransform(indices);
        const TransformResult candidateStats = computeTransformResult(candidateTransform);

        if (candidateStats.totalError < bestStats.totalError)
        {
            bestStats = candidateStats;
            bestTransform = candidateTransform;
            //cout << "New inlier count: " << bestStats.inlierCount << ", " << bestStats.totalError << endl;
        }

        if (ransacIter == constants::RANSACEarlyIters)
        {
            if (bestStats.inlierCount >= constants::RANSACEarlyInlierMin)
                break;
            else
            {
                //cout << "Best inlier count insufficient, running more RANSAC: " << bestStats.inlierCount << endl;
            }
        }
    }

    //
    // recompute transform with all inliers
    //
    transformOutliers = 0;
    transformInliers = 0;
    double inlierDistSum = 0.0;
    set<int> inlierIndices;
    for (const auto &c : iterate(allCorr))
    {
        const vec3f bPt = bestTransform * c.value.ptALocal;
        const float distSq = vec3f::distSq(bPt, c.value.ptBLocal);
        if (distSq < constants::outlierDistSq)
        {
            transformInliers++;
            inlierDistSum += sqrtf(distSq);
            inlierIndices.insert((int)c.index);
            inlierCorr.push_back(c.value);
        }
        else
        {
            transformOutliers++;
        }
    }
    transformInlierError = inlierDistSum / (double)transformInliers;

    if (transformInliers < 4)
    {
        //cout << "Too few inliers to compute transform (" << transformInliers << "): " << imageA->index << "-" << imageB->index << endl;
        //cout << bestTransform << endl;
        //cout << "Best inliers: " << bestStats.inlierCount << endl;
        //visualize(constants::debugDir + to_string(imageA->index) + "_" + to_string(imageB->index) + "/");
        //cin.get();
    }
    else
        transformAToB = estimateTransform(inlierIndices);

    //cout << "inliers: " << transformInliers << " / " << allCorr.size() << ", error: " << transformInlierError << endl;
}

mat4f ImagePairCorrespondences::estimateTransform(const set<int> &indices)
{
    vec3f eigenvalues;
    vector<vec3f> source, target;
    for (int i : indices)
    {
        source.push_back(allCorr[i].ptALocal);
        target.push_back(allCorr[i].ptBLocal);
    }
    const mat4f result = EigenWrapperf::kabsch(source, target, eigenvalues);

    /*const bool check = true;
    if (check && indices.size() == 3)
    {
        for (int i : indices)
        {
            const vec3f aToB = result * allCorr[i].ptALocal;
            const float dist = vec3f::dist(aToB, allCorr[i].ptBLocal);
            if (dist >= 0.04)
            {
                cout << "unexpected dist: " << dist << endl;
                cin.get();
            }
        }
    }*/

    return result;
}

ImagePairCorrespondences::TransformResult ImagePairCorrespondences::computeTransformResult(const mat4f &transform)
{
    TransformResult result;
    result.outlierCount = 0;
    result.inlierCount = 0;
    double inlierDistSum = 0.0;
    for (const ImageCorrespondence &c : allCorr)
    {
        const vec3f bPt = transform * c.ptALocal;
        const float distSq = vec3f::distSq(bPt, c.ptBLocal);
        if (distSq < constants::outlierDistSq)
        {
            result.inlierCount++;
            inlierDistSum += sqrtf(distSq);
        }
        else
        {
            result.outlierCount++;
        }
    }
    
    /*if (result.inlierCount == 0)
    {
        cout << "No inliers?" << endl;
        for (const ImageCorrespondence &c : allCorr)
        {
            const vec3f bPt = transform * c.ptALocal;
            const float dist = vec3f::dist(bPt, c.ptBLocal);
            cout << dist << " " << c.ptALocal << " " << c.ptBLocal << " " << endl;
        }
        cin.get();
    }*/

    if (result.inlierCount == 0)
    {
        result.inlierError = std::numeric_limits<double>::max();
        result.totalError = std::numeric_limits<double>::max();
        return result;
    }

    result.inlierError = inlierDistSum / (double)result.inlierCount;
    result.totalError = result.inlierError + (double)result.outlierCount;
    return result;
}


#include "main.h"

#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

struct FeatureExtractorImpl
{
    FeatureExtractorImpl()
    {
        surf = SURF::create(constants::SIFTHessian);
        freak = FREAK::create();
    }
    Ptr<Feature2D> surf;
    Ptr<Feature2D> freak;
};

FeatureExtractor::FeatureExtractor()
{
    impl = new FeatureExtractorImpl;
}

vector<Keypoint> FeatureExtractor::detectAndDescribe(const Bitmap &bmp)
{
    Mat cvImage(bmp.getDimY(), bmp.getDimX(), CV_8UC1);
    for (const auto &p : bmp)
    {
        const BYTE c = util::boundToByte(((float)p.value.x + (float)p.value.y + (float)p.value.z) / 3.0f);
        cvImage.at<BYTE>((int)p.y, (int)p.x) = c;
    }

    vector<cv::KeyPoint> cvPts;
    impl->surf->detect(cvImage, cvPts);

    Mat descriptors;
    impl->freak->compute(cvImage, cvPts, descriptors);
    const int keyPtCount = (int)cvPts.size();
    
    vector<Keypoint> result(keyPtCount);
    for (int kIndex = 0; kIndex < keyPtCount; kIndex++)
    {
        const auto &k = cvPts[kIndex];
        Keypoint &keypt = result[kIndex];
        keypt.angle = k.angle;
        keypt.response = k.response;
        keypt.size = k.size;
        keypt.pt.x = k.pt.x;
        keypt.pt.y = k.pt.y;
        keypt.color = bmp(math::round(keypt.pt));
        memcpy(keypt.desc.data, descriptors.ptr(kIndex), 64);
    }
    return result;
}

vector<KeypointMatch> KeypointMatcher::match(const vector<Keypoint> &keypointsA, const vector<Keypoint> &keypointsB)
{
    const int aCount = (int)keypointsA.size();
    const int bCount = (int)keypointsB.size();

    vector<KeypointMatch> result;
    for (int a = 0; a < aCount; a++)
    {
        int bestBMatch = findBestIndex(keypointsB, keypointsA[a]);
        int checkAMatch = findBestIndex(keypointsA, keypointsB[bestBMatch]);
        if (checkAMatch == a)
        {
            KeypointMatch match;
            match.indexA = a;
            match.indexB = bestBMatch;
            match.distance = (float)FREAKDescriptor::hammingDistance(keypointsA[a].desc, keypointsB[bestBMatch].desc);
            result.push_back(match);
        }
    }
    
    sort(result.begin(), result.end());
    /*for (auto &r : result)
    {
        cout << r.distance << endl;
    }*/

    return result;
}

int KeypointMatcher::findBestIndex(const vector<Keypoint> &keypoints, const Keypoint &query)
{
    int bestIndex = -1;
    int bestDist = numeric_limits<int>::max();
    for (int i = 0; i < keypoints.size(); i++)
    {
        const int dist = FREAKDescriptor::hammingDistance(keypoints[i].desc, query.desc);
        if (dist < bestDist)
        {
            bestDist = dist;
            bestIndex = i;
        }
    }
    return bestIndex;
}

int FREAKDescriptor::hammingDistance(const FREAKDescriptor &a, const FREAKDescriptor &b)
{
    const UINT64 *aStart = (const UINT64 *)a.data;
    const UINT64 *bStart = (const UINT64 *)b.data;
    int sum = 0;
    for (int i = 0; i < 8; i++)
    {
        sum += helper::countBits(aStart[i] ^ bStart[i]);
    }
    return sum;
}


struct FeatureExtractorImpl;

struct FREAKDescriptor
{
    static int hammingDistance(const FREAKDescriptor &a, const FREAKDescriptor &b);
    BYTE data[64];
};

struct Keypoint
{
    vec2f pt;
    vec4uc color;
    float size;
    float angle;
    float response;

    FREAKDescriptor desc;
};

struct KeypointMatch
{
    int indexA;
    int indexB;
    float distance;
};

inline bool operator < (const KeypointMatch &a, const KeypointMatch &b)
{
    return a.distance < b.distance;
}

class FeatureExtractor
{
public:
    FeatureExtractor();
    vector<Keypoint> detectAndDescribe(const Bitmap &bmp);

    FeatureExtractorImpl *impl;
};

class KeypointMatcher
{
public:
    vector<KeypointMatch> match(const vector<Keypoint> &keypointsA, const vector<Keypoint> &keypointsB);
    int findBestIndex(const vector<Keypoint> &keypoints, const Keypoint &query);
};
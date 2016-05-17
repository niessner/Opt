
const bool useSquashFunction = false;

struct CorrespondenceCostFunc
{
    CorrespondenceCostFunc(const ImageCorrespondence &_corr)
        : corr(_corr) {}

    template <typename T> T squash(const T &x) const
    {
        const T a = (T)20.0;
        const T s = (T)0.5;
        return (( T(2.0) + x / a) / (T(1.0) + exp(T(-1.0) * abs(x * a))) - T(1.0)) * s;
    }

    template <typename T>
    bool operator()(const T* const cameraA, const T* const cameraB, T* residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation
        // camera[3,4,5] are the translation
        
        T pA[3] = { T(corr.ptALocal.x), T(corr.ptALocal.y), T(corr.ptALocal.z) };
        T pAWorld[3];
        ceres::AngleAxisRotatePoint(cameraA, pA, pAWorld);
        pAWorld[0] += cameraA[3]; pAWorld[1] += cameraA[4]; pAWorld[2] += cameraA[5];

        T pB[3] = { T(corr.ptBLocal.x), T(corr.ptBLocal.y), T(corr.ptBLocal.z) };
        T pBWorld[3];
        ceres::AngleAxisRotatePoint(cameraB, pB, pBWorld);
        pBWorld[0] += cameraB[3]; pBWorld[1] += cameraB[4]; pBWorld[2] += cameraB[5];
        
        // The error is the difference between the predicted and observed position.
        if (useSquashFunction)
        {
            residuals[0] = squash(pAWorld[0] - pBWorld[0]) * T(corr.weight);
            residuals[1] = squash(pAWorld[1] - pBWorld[1]) * T(corr.weight);
            residuals[2] = squash(pAWorld[2] - pBWorld[2]) * T(corr.weight);
        }
        else
        {
            residuals[0] = (pAWorld[0] - pBWorld[0]) * T(corr.weight);
            residuals[1] = (pAWorld[1] - pBWorld[1]) * T(corr.weight);
            residuals[2] = (pAWorld[2] - pBWorld[2]) * T(corr.weight);
        }
        return true;
    }

    static ceres::CostFunction* Create(const ImageCorrespondence &corr)
    {
        auto cFunc = new CorrespondenceCostFunc(corr);

        double cameraA[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        double cameraB[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        double residuals[3] = { 0.0, 0.0, 0.0 };
        (*cFunc)(cameraA, cameraB, residuals);

        if (residuals[0] != residuals[0])
            cout << "invalid residual: " << residuals[0] << endl;

        return (new ceres::AutoDiffCostFunction<CorrespondenceCostFunc, 3, 6, 6>(cFunc));
    }

    ImageCorrespondence corr;
};

struct AnchorCostFunc
{
    AnchorCostFunc(const vec3f &_anchorPoint, float _weight)
        : anchorPoint(_anchorPoint), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation
        // camera[3,4,5] are the translation

        T p[3] = { T(anchorPoint.x), T(anchorPoint.y), T(anchorPoint.z) };
        T pWorld[3];
        ceres::AngleAxisRotatePoint(camera, p, pWorld);
        pWorld[0] += camera[3]; pWorld[1] += camera[4]; pWorld[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (pWorld[0] - T(anchorPoint.x)) * T(weight);
        residuals[1] = (pWorld[1] - T(anchorPoint.y)) * T(weight);
        residuals[2] = (pWorld[2] - T(anchorPoint.z)) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec3f &anchorPoint, float weight)
    {
        auto cFunc = new AnchorCostFunc(anchorPoint, weight);

        double camera[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        double residuals[3] = { 0.0, 0.0, 0.0 };
        (*cFunc)(camera, residuals);

        if (residuals[0] != residuals[0])
            cout << "invalid residual: " << residuals[0] << endl;

        return (new ceres::AutoDiffCostFunction<AnchorCostFunc, 3, 6>(cFunc));
    }

    vec3f anchorPoint;
    float weight;
};

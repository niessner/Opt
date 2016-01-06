
#include "mLibInclude.h"

#ifdef USE_CERES

#include <cuda_runtime.h>
#include <cudaUtil.h>

#include "CUDAImageSolver.h"

#include "OptImageSolver.h"
#include "CeresImageSolver.h"
#include "SFSSolverInput.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

const bool performanceTest = false;

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Solver;
using ceres::Solve;
using namespace std;

const int imageDimX = 128;
const int imageDimY = 128;
const int imagePixelCount = imageDimX * imageDimY;

template<class T>
struct vec3T
{
    vec3T(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    T sqMagintude()
    {
        return x * x + y * y + z * z;
    }
    T operator * (T v)
    {
        return vec3T(v * x, v * y, v * z);
    }
    T operator + (const vec3T &v)
    {
        return vec3T(x + v.x, y + v.y, z + v.z);
    }
    T x, y, z;
};

template<class T>
vec3T<T> ppp(const CeresImageSolver &solver, T *X, int posX, int posY, int offX, int offY)
{
    T d = X[getPixel(posX + offX, posY + offY)];
    T i = offX + posX;
    T j = offY + posY;
    vec3T<T> point;
    point.x = ((i - (T)parameters.ux) / (T)parameters.fx) * d;
    point.y = ((j - (T)parameters.uy) / (T)parameters.fy) * d;
    point.z = d;
    return point;
}

template<class T>
T softSelect(T x, T left, T right)
{

}

template<class T>
vec3T<T> normalAt(const CeresImageSolver &solver, T *X, int posX, int posY, int offX, int offY)
{
    ceres::EulerAnglesToRotationMatrix
    T i = offX + posX;
    T j = offY + posY;
    T f_x = (T)solver.f_x;
    T f_y = (T)solver.f_y;

    T X01 = X[getPixel(posX + offX, posY + offY - 1)];
    T X00 = X[getPixel(posX + offX, posY + offY)];
    T X10 = X[getPixel(posX + offX - 1, posY + offY)];

    T n_x = (X01 * (X00 - X10)) / f_y;
    T n_y = (X10 * (X00 - X01)) / f_x;
    T n_z = (n_x * (u_x - i) / f_x) + (n_y * ((T)solver.u_y - j) / f_y) - (X10 * X01) / (f_x * f_y);
    T sqLength = n_x*n_x + n_y*n_y + n_z*n_z + T(1e-6);

    vec3T<T> result;
    T invMagnitude = (T)1.0 / sqrt(sqLength);
    return vec3T<T>(n_x, n_y, n_z) * invMagnitude;
}

template<class T>
T BBB(const CeresImageSolver &solver, T *X, int posX, int posY, int offX, int offY)
{
    vec3T<T> normal = normalAt(solver, X, posX, posY, offX, offY);
    T n_x = normal.x;
    T n_y = normal.y;
    T n_z = normal.z;
    T L[9];
    for (int i = 0; i < 9; i++)
        L[i] = solver.L[i];

    T lighting = L[0] +
        L[1] * n_y + L[2] * n_z + L[3] * n_x +
        L[4] * n_x*n_y + L[5] * n_y*n_z + L[6] * (-n_x*n_x - n_y*n_y + 2 * n_z*n_z) + L[7] * n_z*n_x + L[8] * (n_x*n_x - n_y*n_y);

    return lighting;
}

float III(const CeresImageSolver &solver, int posX, int posY, int offX, int offY)
{
    float Im00 = solver.Im[solver.getPixel(posX + offX, posY + offY)];
    float Im10 = solver.Im[solver.getPixel(posX + offX - 1, posY + offY)];
    float Im01 = solver.Im[solver.getPixel(posX + offX, posY + offY - 1)];
    return Im00 * 0.5f + 0.25f * (Im10 + Im01);
}

// BOUNDS CHECK!!!
// ad.greater(D_i(x-1,y) + D_i(x,y) + D_i(x,y-1), 0)
// opt.InBounds(0,0,1,1)
template<class T>
T B_I(const CeresImageSolver &solver, T *X, int posX, int posY, int offX, int offY)
{
    T bi = BBB(solver, X, posX, posY, offX, offY) - (T)III(solver, posX, posY, offX, offY);
    return bi;
}

struct DepthConstraintTerm
{
    DepthConstraintTerm(const CeresImageSolver *_solver, vec2i _coord, float _weight)
        : solver(_solver), coord(_coord), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const X, T* residuals) const
    {
        T xVal = X[solver->getPixel(coord.x, coord.y)];
        T dVal = (T)solver->D_i[solver->getPixel(coord.x, coord.y)];
        residuals[0] = (xVal - dVal) * T(weight);
        return true;
    }

    // CHECK!!!
    // ad.greater(D_i(0,0), 0)
    // opt.InBounds(0,0,0,0)
    static ceres::CostFunction* Create(const CeresImageSolver *solver, vec2i coord, float weight)
    {
        return (new ceres::AutoDiffCostFunction<DepthConstraintTerm, 1, imagePixelCount>(
            new DepthConstraintTerm(solver, coord, weight)));
    }

    const CeresImageSolver *solver;
    vec2i coord;
    float weight;
};

/*
local USE_MASK_REFINE 			= true

local USE_DEPTH_CONSTRAINT 		= true
local USE_REGULARIZATION 		= true
local USE_SHADING_CONSTRAINT 	= true

local USE_CRAPPY_SHADING_BOUNDARY = true

local DEPTH_DISCONTINUITY_THRE = 0.01

local posX = W:index()
local posY = H:index()

local E_s = 0.0
local E_p = 0.0
local E_r_h = 0.0
local E_r_v = 0.0
local E_r_d = 0.0
local E_g_v = 0.0
local E_g_h = 0.0
local pointValid = ad.greater(D_i(0,0), 0)

if USE_DEPTH_CONSTRAINT then
local E_p_noCheck = X(0,0) - D_i(0,0)
E_p = ad.select(opt.InBounds(0,0,0,0), ad.select(pointValid, E_p_noCheck, 0.0), 0.0)
end

if USE_SHADING_CONSTRAINT then
if USE_CRAPPY_SHADING_BOUNDARY then
local center_tap = B_I(0,0)
local E_g_h_noCheck = B_I(1,0) --(B(1,0) - I(1,0))
local E_g_v_noCheck = B_I(0,1) --(B(0,1) - I(0,1))

local E_g_h_someCheck = center_tap - E_g_h_noCheck
local E_g_v_someCheck = center_tap - E_g_v_noCheck

if USE_MASK_REFINE then
E_g_h_someCheck = E_g_h_someCheck * edgeMaskR(0,0)
E_g_v_someCheck = E_g_v_someCheck * edgeMaskC(0,0)
end
E_g_h = ad.select(opt.InBounds(0,0,1,1), E_g_h_someCheck, 0.0)
E_g_v = ad.select(opt.InBounds(0,0,1,1), E_g_v_someCheck, 0.0)
--E_g_h = center_tap_noCheck - E_g_h_noCheck
--E_g_v = center_tap_noCheck - E_g_v_noCheck

else
local shading_h_valid = ad.greater(D_i(-1,0) + D_i(0,0) + D_i(1,0) + D_i(0,-1) + D_i(1,-1), 0)

local E_g_h_noCheck = B(0,0) - B(1,0) - (I(0,0) - I(1,0))
if USE_MASK_REFINE then
E_g_h_noCheck = E_g_h_noCheck * edgeMaskR(0,0)
end
E_g_h = ad.select(opt.InBounds(0,0,1,1), ad.select(shading_h_valid, E_g_h_noCheck, 0.0), 0.0)

local shading_v_valid = ad.greater(D_i(0,-1) + D_i(0,0) + D_i(0,1) + D_i(-1,0) + D_i(-1,1), 0)

local E_g_v_noCheck = B(0,0) - B(0,1) - (I(0,0) - I(0,1))
if USE_MASK_REFINE then
E_g_v_noCheck = E_g_v_noCheck * edgeMaskC(0,0)
end
E_g_v = ad.select(opt.InBounds(0,0,1,1), ad.select(shading_v_valid, E_g_v_noCheck, 0.0), 0.0)
end
end

local function allpositive(a,...)
local r = ad.greater(a,0)
for i = 1,select("#",...) do
local e = select(i,...)
r = ad.and_(r,ad.greater(e,0))
end
return r
end

if USE_REGULARIZATION then
local cross_valid = allpositive(D_i(0,0), D_i(0,-1) , D_i(0,1) , D_i(-1,0) , D_i(1,0))
local E_s_noCheck = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1))

local d = X(0,0)

local E_s_guard =   ad.and_(ad.less(ad.abs(d - X(0,-1)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(0,1)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(-1,0)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(1,0)), DEPTH_DISCONTINUITY_THRE),
ad.and_(opt.InBounds(0,0,1,1), cross_valid)
))))

E_s = ad.select(E_s_guard,E_s_noCheck,0)
end

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p)

P:Exclude(ad.not_(ad.greater(D_i(0,0),0)))

return P:Cost(cost)
*/
void CeresImageSolver::solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput)
{
    if (width != imageDimX || height != imageDimY)
    {
        cout << "Set imageDims to: " << width << "," << height << endl;
        return;
    }

    const int pixelCount = width * height;
    Xfloat = (float *)result->data();
    D_i = (float *)rawSolverInput.targetDepth->data();
    Im = (float *)rawSolverInput.targetIntensity->data();
    D_p = (float *)rawSolverInput.previousDepth->data();
    edgeMaskR = (BYTE *)rawSolverInput.maskEdgeMap->data();
    edgeMaskC = (BYTE *)rawSolverInput.maskEdgeMap->data() + pixelCount;

    w_p = sqrtf(rawSolverInput.parameters.weightFitting);
    w_s = sqrtf(rawSolverInput.parameters.weightRegularizer);
    w_r = sqrtf(rawSolverInput.parameters.weightPrior);
    w_g = sqrtf(rawSolverInput.parameters.weightShading);
    
    weightShadingStart = rawSolverInput.parameters.weightShadingStart;
    weightShadingIncrement = rawSolverInput.parameters.weightShadingIncrement;
    weightBoundary = rawSolverInput.parameters.weightBoundary;

    f_x = rawSolverInput.parameters.fx;
    f_y = rawSolverInput.parameters.fy;
    u_x = rawSolverInput.parameters.ux;
    u_y = rawSolverInput.parameters.uy;

    deltaTransform = rawSolverInput.parameters.deltaTransform;
    for (int i = 0; i < 9; i++)
        L[i] = rawSolverInput.parameters.lightingCoefficients[i];
    
    auto getPixel = [=](int x, int y) {
        return y * width + x;
    };
    
    ceres::Problem problem;

    double *Xdouble = new double[pixelCount];
    for (int i = 0; i < pixelCount; i++)
    {
        Xdouble[i] = Xfloat[i];
    }

    // add all fit constraints
    //if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
    //    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const bool depthCheck = (D_i[getPixel(x, y)] > 0.0f);
            if (depthCheck)
            {
                ceres::CostFunction* costFunction = DepthConstraintTerm::Create(this, vec2i(x, y), w_p);
                //double2 *varStart = h_x_double + getPixel(x, y);
                problem.AddResidualBlock(costFunction, NULL, Xdouble);
            }
        }
    }

    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = !performanceTest;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::LinearSolverType::CGNR;
    //options.min_linear_solver_iterations = linearIterationMin;
    options.max_num_iterations = 10000;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;

    //options.min_lm_diagonal = 1.0f;
    //options.min_lm_diagonal = options.max_lm_diagonal;
    //options.max_lm_diagonal = 10000000.0;

    //problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    //cout << "Cost*2 start: " << cost << endl;

    double elapsedTime;
    {
        ml::Timer timer;
        Solve(options, &problem, &summary);
        elapsedTime = timer.getElapsedTimeMS();
    }

    cout << "Solver used: " << summary.linear_solver_type_used << endl;
    cout << "Minimizer iters: " << summary.iterations.size() << endl;
    cout << "Total time: " << elapsedTime << "ms" << endl;

    double iterationTotalTime = 0.0;
    int totalLinearItereations = 0;
    for (auto &i : summary.iterations)
    {
        iterationTotalTime += i.iteration_time_in_seconds;
        totalLinearItereations += i.linear_solver_iterations;
        cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms" << endl;
    }

    cout << "Total iteration time: " << iterationTotalTime << endl;
    cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

    double cost = -1.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 end: " << cost * 2 << endl;

    cout << summary.FullReport() << endl;

    for (int i = 0; i < pixelCount; i++)
    {
        Xfloat[i] = (float)Xdouble[i];
    }
}

#endif
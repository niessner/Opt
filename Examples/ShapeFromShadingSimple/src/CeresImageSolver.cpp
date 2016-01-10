
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

template<class T>
struct vec3T
{
    vec3T(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    vec3T operator * (T v)
    {
        return vec3T(v * x, v * y, v * z);
    }
    vec3T operator + (const vec3T &v)
    {
        return vec3T(x + v.x, y + v.y, z + v.z);
    }
    vec3T operator - (const vec3T &v)
    {
        return vec3T(x - v.x, y - v.y, z - v.z);
    }
    T x, y, z;
};

template<class T>
struct SFSStencil
{
    SFSStencil(const T* const Xmm, const T* const X0m, const T* const X1m,
               const T* const Xm0, const T* const X00, const T* const X10,
               const T* const Xm1, const T* const X01, const T* const X11)
    {
        values[0][0] = Xmm[0];
        values[0][1] = Xm0[0];
        values[0][2] = Xm1[0];

        values[1][0] = X0m[0];
        values[1][1] = X00[0];
        values[1][2] = X01[0];

        values[2][0] = X1m[0];
        values[2][1] = X10[0];
        values[2][2] = X11[0];
    }
    T& operator()(int offX, int offY)
    {
        return values[offX + 1][offY + 1];
    }
private:
    T values[3][3];
};

template<class T>
vec3T<T> ppp(const CeresImageSolver &solver, SFSStencil<T> &stencil, int posX, int posY, int offX, int offY)
{
    T d = stencil(offX, offY);
    T i = (T)(offX + posX);
    T j = (T)(offY + posY);
    vec3T<T> point(
        ((i - (T)solver.u_x) / (T)solver.f_x) * d,
        ((j - (T)solver.u_y) / (T)solver.f_y) * d,
        d);
    return point;
}

template<class T>
vec3T<T> normalAt(const CeresImageSolver &solver, SFSStencil<T> &stencil, int posX, int posY, int offX, int offY)
{
    T i = (T)(offX + posX);
    T j = (T)(offY + posY);
    T f_x = (T)solver.f_x;
    T f_y = (T)solver.f_y;
    T u_x = (T)solver.u_x;
    T u_y = (T)solver.u_y;

    T X01 = stencil(offX, offY - 1);
    T X00 = stencil(offX, offY);
    T X10 = stencil(offX - 1, offY);
    
    T n_x = (X01 * (X00 - X10)) / f_y;
    T n_y = (X10 * (X00 - X01)) / f_x;
    T n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (X10 * X01 / (f_x * f_y));
    T sqLength = n_x*n_x + n_y*n_y + n_z*n_z + T(1e-20);

    T invMagnitude = (T)1.0 / sqrt(sqLength);
    return vec3T<T>(n_x, n_y, n_z) * invMagnitude;
}

template<class T>
T BBB(const CeresImageSolver &solver, SFSStencil<T> &stencil, int posX, int posY, int offX, int offY)
{
    vec3T<T> normal = normalAt(solver, stencil, posX, posY, offX, offY);
    T n_x = normal.x;
    T n_y = normal.y;
    T n_z = normal.z;
    T L[9];
    for (int i = 0; i < 9; i++)
        L[i] = (T)solver.L[i];

    T lighting = L[0] +
        L[1] * n_y + L[2] * n_z + L[3] * n_x +
        L[4] * n_x*n_y + L[5] * n_y*n_z + L[6] * (-n_x*n_x - n_y*n_y + (T)2.0 * n_z*n_z) + L[7] * n_z*n_x + L[8] * (n_x*n_x - n_y*n_y);

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
T B_I(const CeresImageSolver &solver, SFSStencil<T> &stencil, int posX, int posY, int offX, int offY)
{
    T bi = BBB(solver, stencil, posX, posY, offX, offY) - (T)III(solver, posX, posY, offX, offY);
    //D_i(x-1,y) + D_i(x,y) + D_i(x,y-1)
    float sum = solver.D_i[solver.getPixel(posX + offX - 1, posY + offY)] +
                solver.D_i[solver.getPixel(posX + offX, posY + offY)] +
                solver.D_i[solver.getPixel(posX + offX, posY + offY - 1)];
    if (sum > 0.0f)
        return (T)0.0f;
    return bi;
}

struct DepthConstraintTerm
{
    DepthConstraintTerm(const CeresImageSolver *_solver, vec2i _coord, float _weight)
        : solver(_solver), coord(_coord), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const X00, T* residuals) const
    {
        T dVal = (T)solver->D_i[solver->getPixel(coord.x, coord.y)];
        //T dVal = (T)0.4;
        residuals[0] = (X00[0] - dVal) * T(weight);
        return true;
    }

    // CHECK!!!
    // ad.greater(D_i(0,0), 0)
    static ceres::CostFunction* Create(const CeresImageSolver *solver, vec2i coord, float weight)
    {
        return (new ceres::AutoDiffCostFunction<DepthConstraintTerm, 1, 1>(
            new DepthConstraintTerm(solver, coord, weight)));
    }

    const CeresImageSolver *solver;
    vec2i coord;
    float weight;
};

// guarded by D_i(0,0), D_i(0,-1) , D_i(0,1) , D_i(-1,0) , D_i(1,0) are all > 0
struct RegTerm
{
    RegTerm(const CeresImageSolver *_solver, vec2i _coord, float _weight)
        : solver(_solver), coord(_coord), weight(_weight) {}
    template <typename T>
    bool operator()(const T* const Xmm, const T* const X0m, const T* const X1m,
        const T* const Xm0, const T* const X00, const T* const X10,
        const T* const Xm1, const T* const X01, const T* const X11,
        T* residuals) const
    {
        SFSStencil<T> stencil(Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);
        
        //E_s = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1))
        vec3T<T> p00 = ppp(*solver, stencil, coord.x, coord.y, 0, 0);
        vec3T<T> pm0 = ppp(*solver, stencil, coord.x, coord.y, -1, 0);
        vec3T<T> p0m = ppp(*solver, stencil, coord.x, coord.y, 0, -1);
        vec3T<T> p10 = ppp(*solver, stencil, coord.x, coord.y, 1, 0);
        vec3T<T> p01 = ppp(*solver, stencil, coord.x, coord.y, 0, 1);

        vec3T<T> val = p00 * (T)4.0 - (pm0 + p0m + p10 + p01);
        residuals[0] = val.x * T(weight);
        residuals[1] = val.y * T(weight);
        residuals[2] = val.z * T(weight);
        return true;
    }

    // CHECK!!!
    // allpositive(D_i(0,0), D_i(0,-1) , D_i(0,1) , D_i(-1,0) , D_i(1,0))
    static ceres::CostFunction* Create(const CeresImageSolver *solver, vec2i coord, float weight)
    {
        return (new ceres::AutoDiffCostFunction<RegTerm, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
            new RegTerm(solver, coord, weight)));
    }

    const CeresImageSolver *solver;
    vec2i coord;
    float weight;
};

struct HorzShadingTerm
{
    HorzShadingTerm(const CeresImageSolver *_solver, vec2i _coord, float _weight)
        : solver(_solver), coord(_coord), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const Xmm, const T* const X0m, const T* const X1m,
        const T* const Xm0, const T* const X00, const T* const X10,
        const T* const Xm1, const T* const X01, const T* const X11,
        T* residuals) const
    {
        SFSStencil<T> stencil(Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);

        //E_g_h_someCheck = B_I(0,0) - B_I(1,0)
        T BI00 = B_I(*solver, stencil, coord.x, coord.y, 0, 0);
        T BI10 = B_I(*solver, stencil, coord.x, coord.y, 1, 0);
        residuals[0] = (BI00 - BI10) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const CeresImageSolver *solver, vec2i coord, float weight)
    {
        return (new ceres::AutoDiffCostFunction<HorzShadingTerm, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
            new HorzShadingTerm(solver, coord, weight)));
    }

    const CeresImageSolver *solver;
    vec2i coord;
    float weight;
};

struct VertShadingTerm
{
    VertShadingTerm(const CeresImageSolver *_solver, vec2i _coord, float _weight)
        : solver(_solver), coord(_coord), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const Xmm, const T* const X0m, const T* const X1m,
        const T* const Xm0, const T* const X00, const T* const X10,
        const T* const Xm1, const T* const X01, const T* const X11,
        T* residuals) const
    {
        SFSStencil<T> stencil(Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);

        //E_g_h_someCheck = B_I(0,0) - B_I(1,0)
        T BI00 = B_I(*solver, stencil, coord.x, coord.y, 0, 0);
        T BI01 = B_I(*solver, stencil, coord.x, coord.y, 0, 1);
        residuals[0] = (BI00 - BI01) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const CeresImageSolver *solver, vec2i coord, float weight)
    {
        return (new ceres::AutoDiffCostFunction<VertShadingTerm, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
            new VertShadingTerm(solver, coord, weight)));
    }

    const CeresImageSolver *solver;
    vec2i coord;
    float weight;
};

/*
local DEPTH_DISCONTINUITY_THRE = 0.01

local posX = W:index()
local posY = H:index()

local E_s = 0.0
local E_r_h = 0.0
local E_r_v = 0.0
local E_r_d = 0.0
local E_g_v = 0.0
local E_g_h = 0.0
local pointValid = ad.greater(D_i(0,0), 0)

local center_tap = B_I(0,0)
local E_g_h_noCheck = B_I(1,0)
local E_g_v_noCheck = B_I(0,1)

local E_g_h_someCheck = center_tap - E_g_h_noCheck
local E_g_v_someCheck = center_tap - E_g_v_noCheck

E_g_h_someCheck = E_g_h_someCheck * edgeMaskR(0,0)
E_g_v_someCheck = E_g_v_someCheck * edgeMaskC(0,0)

E_g_h = ad.select(opt.InBounds(0,0,1,1), E_g_h_someCheck, 0.0)
E_g_v = ad.select(opt.InBounds(0,0,1,1), E_g_v_someCheck, 0.0)

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p)

P:Exclude(ad.not_(ad.greater(D_i(0,0),0)))

return P:Cost(cost)
*/
void CeresImageSolver::solve(std::shared_ptr<SimpleBuffer> result, const SFSSolverInput& rawSolverInput)
{
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

    const bool useFitConstraint = true;
    const bool useRegConstraint = true;
    const bool useHorzConstraint = false;
    const bool useVertConstraint = false;

    const int borderSize = 1;
    if (useFitConstraint)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                const bool depthCheck = (D_i[getPixel(x, y)] > 0.0f);
                if (depthCheck)
                {
                    ceres::CostFunction* costFunction = DepthConstraintTerm::Create(this, vec2i(x, y), w_p);
                    double *X00 = Xdouble + getPixel(x + 0, y + 0);
                    problem.AddResidualBlock(costFunction, NULL, X00);
                }
            }
        }
    }

    if (useRegConstraint)
    {
        for (int y = borderSize; y < height - borderSize; y++)
        {
            for (int x = borderSize; x < width - borderSize; x++)
            {
                const bool depthCheck = (D_i[getPixel(x + 0, y + 0)] > 0.0f &&
                                         D_i[getPixel(x + 1, y + 0)] > 0.0f &&
                                         D_i[getPixel(x - 1, y + 0)] > 0.0f &&
                                         D_i[getPixel(x + 0, y + 1)] > 0.0f &&
                                         D_i[getPixel(x + 0, y - 1)] > 0.0f);
                if (depthCheck)
                {
                    ceres::CostFunction* costFunction = RegTerm::Create(this, vec2i(x, y), w_s);
                    double *Xmm = Xdouble + getPixel(x - 1, y - 1);
                    double *X0m = Xdouble + getPixel(x + 0, y - 1);
                    double *X1m = Xdouble + getPixel(x + 1, y - 1);
                    double *Xm0 = Xdouble + getPixel(x - 1, y + 0);
                    double *X00 = Xdouble + getPixel(x + 0, y + 0);
                    double *X10 = Xdouble + getPixel(x + 1, y + 0);
                    double *Xm1 = Xdouble + getPixel(x - 1, y + 1);
                    double *X01 = Xdouble + getPixel(x + 0, y + 1);
                    double *X11 = Xdouble + getPixel(x + 1, y + 1);

                    problem.AddResidualBlock(costFunction, NULL, Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);
                }
            }
        }
    }

    if (useHorzConstraint)
    {
        for (int y = borderSize; y < height - borderSize; y++)
        {
            for (int x = borderSize; x < width - borderSize; x++)
            {
                const bool depthCheck = (D_i[getPixel(x + 0, y + 0)] > 0.0f &&
                                         edgeMaskR[getPixel(x + 0, y + 0)] != 0);
                if (depthCheck)
                {
                    ceres::CostFunction* costFunction = HorzShadingTerm::Create(this, vec2i(x, y), w_g);
                    double *Xmm = Xdouble + getPixel(x - 1, y - 1);
                    double *X0m = Xdouble + getPixel(x + 0, y - 1);
                    double *X1m = Xdouble + getPixel(x + 1, y - 1);
                    double *Xm0 = Xdouble + getPixel(x - 1, y + 0);
                    double *X00 = Xdouble + getPixel(x + 0, y + 0);
                    double *X10 = Xdouble + getPixel(x + 1, y + 0);
                    double *Xm1 = Xdouble + getPixel(x - 1, y + 1);
                    double *X01 = Xdouble + getPixel(x + 0, y + 1);
                    double *X11 = Xdouble + getPixel(x + 1, y + 1);

                    problem.AddResidualBlock(costFunction, NULL, Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);
                }
            }
        }
    }

    if (useVertConstraint)
    {
        for (int y = borderSize; y < height - borderSize; y++)
        {
            for (int x = borderSize; x < width - borderSize; x++)
            {
                const bool depthCheck = (D_i[getPixel(x + 0, y + 0)] > 0.0f &&
                    edgeMaskC[getPixel(x + 0, y + 0)] != 0);
                if (depthCheck)
                {
                    ceres::CostFunction* costFunction = VertShadingTerm::Create(this, vec2i(x, y), w_g);
                    double *Xmm = Xdouble + getPixel(x - 1, y - 1);
                    double *X0m = Xdouble + getPixel(x + 0, y - 1);
                    double *X1m = Xdouble + getPixel(x + 1, y - 1);
                    double *Xm0 = Xdouble + getPixel(x - 1, y + 0);
                    double *X00 = Xdouble + getPixel(x + 0, y + 0);
                    double *X10 = Xdouble + getPixel(x + 1, y + 0);
                    double *Xm1 = Xdouble + getPixel(x - 1, y + 1);
                    double *X01 = Xdouble + getPixel(x + 0, y + 1);
                    double *X11 = Xdouble + getPixel(x + 1, y + 1);

                    problem.AddResidualBlock(costFunction, NULL, Xmm, X0m, X1m, Xm0, X00, X10, Xm1, X01, X11);
                }
            }
        }
    }

    double cost = -1.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 start: " << cost * 2 << endl;

    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = !performanceTest;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    //options.linear_solver_type = ceres::LinearSolverType::CGNR;
    //options.min_linear_solver_iterations = linearIterationMin;
    options.max_num_iterations = 10000;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;

    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-4 * options.function_tolerance;

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

    cost = -1.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 end: " << cost * 2 << endl;

    cout << summary.FullReport() << endl;

    for (int i = 0; i < pixelCount; i++)
    {
        Xfloat[i] = (float)Xdouble[i];
    }
}

#endif
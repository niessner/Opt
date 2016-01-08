
#pragma once

#include "main.h"

#ifdef USE_CERES

const bool performanceTest = true;
//const int linearIterationMin = 100;

#include <cuda_runtime.h>

#include "OpenMesh.h"
#include "CeresWarpingSolver.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

template<class T>
struct vec3T
{
    vec3T() {}
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
    vec3T operator * (T v)
    {
        return vec3T(v * x, v * y, v * z);
    }
    vec3T operator + (const vec3T &v)
    {
        return vec3T(x + v.x, y + v.y, z + v.z);
    }
    const T& operator [](int k) const
    {
        if (k == 0) return x;
        if (k == 1) return y;
        if (k == 2) return z;
        return x;
    }
    T x, y, z;
};

template<class T>
void evalRot(T CosAlpha, T CosBeta, T CosGamma, T SinAlpha, T SinBeta, T SinGamma, T R[9])
{
    R[0] = CosGamma*CosBeta;
    R[1] = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
    R[2] = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
    R[3] = SinGamma*CosBeta;
    R[4] = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
    R[5] = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
    R[6] = -SinBeta;
    R[7] = CosBeta*SinAlpha;
    R[8] = CosBeta*CosAlpha;
}

template<class T>
void evalR(T alpha, T beta, T gamma, T R[9])
{
    evalRot(cos(alpha), cos(beta), cos(gamma), sin(alpha), sin(beta), sin(gamma), R);
}

template<class T>
vec3T<T> mul(T matrix[9], const vec3T<T> &v)
{
    vec3T<T> result;
    result.x = matrix[0]*v[0] + matrix[1]*v[1] + matrix[2]*v[2];
    result.y = matrix[3]*v[0] + matrix[4]*v[1] + matrix[5]*v[2];
    result.z = matrix[6]*v[0] + matrix[7]*v[1] + matrix[8]*v[2];
    return result;
}

/*
local P = adP.P
local W,H = opt.Dim("W",0), opt.Dim("H",1)

local X = 			adP:Image("X", opt.float6,W,H,0)			--vertex.xyz, rotation.xyz <- unknown
local UrShape = 	adP:Image("UrShape", opt.float3,W,H,1)		--urshape: vertex.xyz
local Constraints = adP:Image("Constraints", opt.float3,W,H,2)	--constraints
local G = adP:Graph("G", 0, "v0", W, H, 0, "v1", W, H, 1)
P:Stencil(2)
P:UsePreconditioner(true)

--regularization
local x0 = X(G.v0)	--float6
local x1 = X(G.v1)	--float6
local x = ad.Vector(x0(0), x0(1), x0(2))	--vertex-unknown : float3
local a = ad.Vector(x0(3), x0(4), x0(5))	--rotation(alpha,beta,gamma) : float3
local R = evalR(a(0), a(1), a(2))			--rotation : float3x3
local xHat = UrShape(G.v0)					--uv-urshape : float3

local n = ad.Vector(x1(0), x1(1), x1(2))
local ARAPCost = (x - n) - mul(R, (xHat - UrShape(G.v1)))

terms:insert(w_regSqrt*ARAPCost)
*/

struct FitTerm
{
    FitTerm(const vec3f &_constraint, float _weight)
        : constraint(_constraint), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const positions, T* residuals) const
    {
        residuals[0] = (positions[0] - T(constraint.x)) * T(weight);
        residuals[1] = (positions[1] - T(constraint.y)) * T(weight);
        residuals[2] = (positions[2] - T(constraint.z)) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec3f &constraint, float weight)
    {
        return (new ceres::AutoDiffCostFunction<FitTerm, 3, 3>(
            new FitTerm(constraint, weight)));
    }

    vec3f constraint;
    float weight;
};

struct RegTerm
{
    RegTerm(const vec3f &_deltaUr, float _weight)
        : deltaUr(_deltaUr), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
    {
        //cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
        T R[9];
        evalR(a[0], a[1], a[2], R);

        vec3T<T> urOut = mul(R, vec3T<T>(T(deltaUr.x), T(deltaUr.y), T(deltaUr.z)));
        
        T deltaX[3];
        deltaX[0] = xA[0] - xB[0];
        deltaX[1] = xA[1] - xB[1];
        deltaX[2] = xA[2] - xB[2];

        residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
        residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
        residuals[2] = (deltaX[2] - urOut[2]) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec3f &deltaUr, float weight)
    {
        return (new ceres::AutoDiffCostFunction<RegTerm, 3, 3, 3, 3>(
            new RegTerm(deltaUr, weight)));
    }

    vec3f deltaUr;
    float weight;
};

vec3f toVec(const float3 &v)
{
    return vec3f(v.x, v.y, v.z);
}

float CeresWarpingSolver::solveGN(
    float3* vertexPosFloat3,
    float3* anglesFloat3,
    float3* vertexPosFloat3Urshape,
    float3* vertexPosTargetFloat3,
    float weightFit,
    float weightReg)
{
    float weightFitSqrt = sqrt(weightFit);
    float weightRegSqrt = sqrt(weightReg);

    for (int i = 0; i < vertexCount; i++)
    {
        vertexPosDouble3[i].x = vertexPosFloat3[i].x;
        vertexPosDouble3[i].y = vertexPosFloat3[i].y;
        vertexPosDouble3[i].z = vertexPosFloat3[i].z;
        anglesDouble3[i].x = anglesFloat3[i].x;
        anglesDouble3[i].y = anglesFloat3[i].y;
        anglesDouble3[i].z = anglesFloat3[i].z;
    }

    Problem problem;

    // add all fit constraints
    //if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
    //    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
    for (auto v = mesh->vertices_begin(); v != mesh->vertices_end(); v++)
    {
        int myIndex = v->idx();
        

        const vec3f constraint = toVec(vertexPosTargetFloat3[myIndex]);
        if (constraint.x > -999999.0f)
        {
            ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
            double3 *varStart = vertexPosDouble3 + myIndex;
            problem.AddResidualBlock(costFunction, NULL, (double *)varStart);
        }
    }

    for (auto v = mesh->vertices_begin(); v != mesh->vertices_end(); v++)
    {
        unsigned int valence = mesh->valence(*v);
        for (auto vv = mesh->vv_iter(*v); vv.is_valid(); vv++)
        {
            auto handleNeighbor(*vv);

            int myIndex = v->idx();
            int neighborIndex = handleNeighbor.idx();

            vec3f deltaUr = toVec(vertexPosFloat3Urshape[myIndex]) - toVec(vertexPosFloat3Urshape[neighborIndex]);
            ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
            double3 *varStartA = vertexPosDouble3 + myIndex;
            double3 *varStartB = vertexPosDouble3 + neighborIndex;
            double3 *angleStartA = anglesDouble3 + myIndex;
            problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, (double*)angleStartA);
        }
    }

    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = !performanceTest;

    //faster methods
    options.num_threads = 1;
    options.num_linear_solver_threads = 1;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s

    //slower methods
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
    //options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s

    //options.min_linear_solver_iterations = linearIterationMin;
    options.max_num_iterations = 10000;
    options.function_tolerance = 0.2;
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

    double cost = -1.0;
    problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 end: " << cost * 2 << endl;

    cout << summary.FullReport() << endl;

    for (int i = 0; i < vertexCount; i++)
    {
        vertexPosFloat3[i].x = (float)vertexPosDouble3[i].x;
        vertexPosFloat3[i].y = (float)vertexPosDouble3[i].y;
        vertexPosFloat3[i].z = (float)vertexPosDouble3[i].z;
        anglesFloat3[i].x = (float)anglesDouble3[i].x;
        anglesFloat3[i].y = (float)anglesDouble3[i].y;
        anglesFloat3[i].z = (float)anglesDouble3[i].z;
    }

    return (float)(summary.total_time_in_seconds * 1000.0);
}

#endif

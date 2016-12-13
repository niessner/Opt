#pragma once

#include "main.h"

#ifdef USE_CERES
const bool performanceTest = true;
//const int linearIterationMin = 100;

#include <cuda_runtime.h>
#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"
#include "CeresSolverImageWarping.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

//--fitting
//local constraintUV = Constraints(0, 0)	--float2
//
//local e_fit = ad.select(ad.eq(m, 0.0), x - constraintUV, ad.Vector(0.0, 0.0))
//e_fit = ad.select(ad.greatereq(constraintUV(0), 0.0), e_fit, ad.Vector(0.0, 0.0))
//e_fit = ad.select(ad.greatereq(constraintUV(1), 0.0), e_fit, ad.Vector(0.0, 0.0))
//
//terms:insert(w_fitSqrt*e_fit)

/*
Translation A:
if(mask(i, j) == 0)
    fit = x - constraints(i, j)
else
    fit = 0

if(constaints(i, j).u < 0 || constaints(i, j).v < 0)
    fit = 0

Translation B:
if(mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
*/

//--reg
//local xHat = UrShape(0,0)
//local i, j = unpack(o)
//local n = ad.Vector(X(i, j, 0), X(i, j, 1))
//local ARAPCost = (x - n) - mul(R, (xHat - UrShape(i, j)))
//local ARAPCostF = ad.select(opt.InBounds(0, 0, 0, 0), ad.select(opt.InBounds(i, j, 0, 0), ARAPCost, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
//local m = Mask(i, j)
//ARAPCostF = ad.select(ad.eq(m, 0.0), ARAPCostF, ad.Vector(0.0, 0.0))
//terms:insert(w_regSqrt*ARAPCostF)

/*
Translation A:

// ox,oy = offset x/y
if(mask(i, j) == 0 && offset-in-bounds)
    cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
*/

vec2f toVec(const OPT_FLOAT2 &v)
{
    return vec2f(v.x, v.y);
}


template<class T> void evalR(const T angle, T matrix[4])
{
    T cosAlpha = cos(angle);
    T sinAlpha = sin(angle);
    matrix[0] = cosAlpha;
    matrix[1] = -sinAlpha;
    matrix[2] = sinAlpha;
    matrix[3] = cosAlpha;
}

template<class T> void mul(const T matrix[4], const T vx, const T vy, T out[2])
{
    out[0] = matrix[0] * vx + matrix[1] * vy;
    out[1] = matrix[2] * vx + matrix[3] * vy;
}

struct FitTerm
{
    FitTerm(const vec2f &_constraint, float _weight)
        : constraint(_constraint), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const x, T* residuals) const
    {
        residuals[0] = (x[0] - T(constraint.x)) * T(weight);
        residuals[1] = (x[1] - T(constraint.y)) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec2f &constraint, float weight)
    {
        return (new ceres::AutoDiffCostFunction<FitTerm, 2, 2>(
            new FitTerm(constraint, weight)));
    }

    vec2f constraint;
    float weight;
};

struct RegTerm
{
    RegTerm(const vec2f &_deltaUr, float _weight)
        : deltaUr(_deltaUr), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
    {
        //cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
        T R[4];
        evalR(a[0], R);

        T urOut[2];
        mul(R, (T)deltaUr.x, (T)deltaUr.y, urOut);

        T deltaX[2];
        deltaX[0] = xA[0] - xB[0];
        deltaX[1] = xA[1] - xB[1];

        residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
        residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec2f &deltaUr, float weight)
    {
        return (new ceres::AutoDiffCostFunction<RegTerm, 2, 2, 2, 1>(
            new RegTerm(deltaUr, weight)));
    }

    vec2f deltaUr;
    float weight;
};

float CeresSolverWarping::solve(OPT_FLOAT2* h_x_float, OPT_FLOAT* h_a_float, OPT_FLOAT2* h_urshape, OPT_FLOAT2* h_constraints, OPT_FLOAT* h_mask, float weightFit, float weightReg, std::vector<SolverIteration>& result)
{
    float weightFitSqrt = sqrt(weightFit);
    float weightRegSqrt = sqrt(weightReg);

    Problem problem;

    auto getPixel = [=](int x, int y) {
        return y * m_width + x;
    };

    const int pixelCount = m_width * m_height;
    for (int i = 0; i < pixelCount; i++)
    {
        h_x_double[i].x = h_x_float[i].x;
        h_x_double[i].y = h_x_float[i].y;
        h_a_double[i] = h_a_float[i];
    }

    // add all fit constraints
    //if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
    //    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
    for (int y = 0; y < m_height; y++)
    {
        for (int x = 0; x < m_width; x++)
        {
            float mask = h_mask[getPixel(x, y)];
            const vec2f constraint = toVec(h_constraints[getPixel(x, y)]);
            if (mask == 0.0f && constraint.x >= 0.0f && constraint.y >= 0.0f)
            {
                ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
                double2 *varStart = h_x_double + getPixel(x, y);
                problem.AddResidualBlock(costFunction, NULL, (double*)varStart);
            }
        }
    }

    //add all reg constraints
    //if(mask(i, j) == 0 && offset-in-bounds)
    //  cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
    for (int y = 0; y < m_height; y++)
    {
        for (int x = 0; x < m_width; x++)
        {
            float mask = h_mask[getPixel(x, y)];
            if (mask == 0.0f) {
                const vec2i offsets[] = { vec2i(0, 1), vec2i(1, 0), vec2i(0, -1), vec2i(-1, 0) };
                for (vec2i offset : offsets)
                {
                    const vec2i oPos = offset + vec2i(x, y);
                    float innerMask = h_mask[getPixel(oPos.x, oPos.y)];
                    if (innerMask == 0.0f && oPos.x >= 0 && oPos.x < m_width && oPos.y >= 0 && oPos.y < m_height)
                    {
                        vec2f deltaUr = toVec(h_urshape[getPixel(x, y)]) - toVec(h_urshape[getPixel(oPos.x, oPos.y)]);
                        ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
                        double2 *varStartA = h_x_double + getPixel(x, y);
                        double2 *varStartB = h_x_double + getPixel(oPos.x, oPos.y);
                        problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, h_a_double + getPixel(x, y));
                    }
                }
            } 
        }
    }
    
    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

   // options.minimizer_progress_to_stdout = true;// !performanceTest;

    options.minimizer_progress_to_stdout = true;

    //faster methods
    options.num_threads = 12;
    options.num_linear_solver_threads = 12;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s


    options.max_num_iterations = 1000;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s
    
    //slower methods
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
    //options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s
    
    //options.minimizer_type = ceres::LINE_SEARCH;

    //options.min_linear_solver_iterations = linearIterationMin;
    
    //options.function_tolerance = 0.01;
    //options.gradient_tolerance = 1e-4 * options.function_tolerance;

    //options.min_lm_diagonal = 1.0f;
    //options.min_lm_diagonal = options.max_lm_diagonal;
    //options.max_lm_diagonal = 10000000.0;

    //problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    //cout << "Cost*2 start: " << cost << endl;

    // TODO: remove
    //options.linear_solver_type = ceres::LinearSolverType::CGNR; 


    options.function_tolerance = 1e-20;
    options.gradient_tolerance = 1e-10 * options.function_tolerance;

    // Default values, reproduced here for clarity
    //options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    options.max_trust_region_radius = 1e16;
    options.min_trust_region_radius = 1e-32;
    options.min_relative_decrease = 1e-3;
    // Disable to match Opt
    //options.min_lm_diagonal = 1e-32;
    //options.max_lm_diagonal = std::numeric_limits<double>::infinity();
    //options.min_trust_region_radius = 1e-256;

    //options.initial_trust_region_radius = 0.005;

    options.initial_trust_region_radius = 1e4;
    //options.initial_trust_region_radius = 1e7;
    //options.max_linear_solver_iterations = 20;
    
    options.eta = 1e-4;

    options.jacobi_scaling = true;
    //options.preconditioner_type = ceres::PreconditionerType::IDENTITY;


    options.max_num_iterations = 100;

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

    for (auto &i : summary.iterations) {
        result.push_back(SolverIteration(i.cost, i.iteration_time_in_seconds * 1000.0));
    }

    cout << "Total iteration time: " << iterationTotalTime << endl;
    cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

    double cost = -1.0;
    problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 end: " << cost * 2 << endl;

    cout << summary.FullReport() << endl;

    for (int i = 0; i < pixelCount; i++)
    {
        h_x_float[i].x = (float)h_x_double[i].x;
        h_x_float[i].y = (float)h_x_double[i].y;
        h_a_float[i] = (float)h_a_double[i];
    }

    return (float)(summary.total_time_in_seconds * 1000.0);
}

#endif
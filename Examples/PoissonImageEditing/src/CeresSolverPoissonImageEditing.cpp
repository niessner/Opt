#pragma once

#include "main.h"

#ifdef USE_CERES

#include <cuda_runtime.h>

#include "CeresSolverPoissonImageEditing.h"
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

vec4f toVec(const float4 &v)
{
    return vec4f(v.x, v.y, v.z, v.w);
}

struct EdgeTerm
{
    EdgeTerm(const vec4f &_targetDelta, float _weight)
        : targetDelta(_targetDelta), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const xA, const T* const xB, T* residuals) const
    {
        for (int i = 0; i < 4; i++)
            residuals[i] = (xA[i] - xB[i] - T(targetDelta[i])) * T(weight);
        /*residuals[0] = xA[0];
        residuals[1] = xA[1];
        residuals[2] = xA[2];
        residuals[3] = xA[3];*/
        return true;
    }

    static ceres::CostFunction* Create(const vec4f &targetDelta, float weight)
    {
        return (new ceres::AutoDiffCostFunction<EdgeTerm, 4, 4, 4>(
            new EdgeTerm(targetDelta, weight)));
    }

    vec4f targetDelta;
    float weight;
};

struct FitTerm
{
    FitTerm(const vec4f &_targetValue, float _weight)
        : targetValue(_targetValue), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const x, T* residuals) const
    {
        for (int i = 0; i < 4; i++)
            residuals[i] = (x[i] - T(targetValue[i])) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec4f &targetValue, float weight)
    {
        return (new ceres::AutoDiffCostFunction<FitTerm, 4, 4>(
            new FitTerm(targetValue, weight)));
    }

    vec4f targetValue;
    float weight;
};

void CeresSolverPoissonImageEditing::solve(float4* h_unknownFloat , float4* h_target, float* h_mask, float weightFit, float weightReg)
{
    float weightFitSqrt = sqrt(weightFit);
    float weightRegSqrt = sqrt(weightReg);

    Problem problem;

    auto getPixel = [=](int x, int y) {
        return y * width + x;
    };

    const int pixelCount = width * height;
    double4 *h_unknownDouble = new double4[pixelCount];
    for (int i = 0; i < pixelCount; i++)
    {
        h_unknownDouble[i].x = h_unknownFloat[i].x;
        h_unknownDouble[i].y = h_unknownFloat[i].y;
        h_unknownDouble[i].z = h_unknownFloat[i].z;
        h_unknownDouble[i].w = h_unknownFloat[i].w;
    }

    vector< pair<int, int> > edges;
    for (int y = 0; y < height - 1; y++)
    {
        for (int x = 0; x < width - 1; x++)
        {
            int pixel00 = getPixel(x + 0, y + 0);
            int pixel10 = getPixel(x + 1, y + 0);
            int pixel01 = getPixel(x + 0, y + 1);
            int pixel11 = getPixel(x + 1, y + 1);
            edges.push_back(make_pair(pixel00, pixel10));
            edges.push_back(make_pair(pixel00, pixel01));

            edges.push_back(make_pair(pixel11, pixel10));
            edges.push_back(make_pair(pixel11, pixel01));
        }
    }

    cout << "Edges: " << edges.size() << endl;

    int edgesAdded = 0;
    // add all edge constraints
    for (auto &e : edges)
    {
        const float mask = h_mask[e.first];
        if (mask == 0.0f)
        {
            const vec4f targetA = toVec(h_target[e.first]);
            const vec4f targetB = toVec(h_target[e.second]);

            vec4f targetDelta = targetA - targetB;
            ceres::CostFunction* costFunction = EdgeTerm::Create(targetA - targetB, 1.0f);
            double4 *varStartA = h_unknownDouble + e.first;
            double4 *varStartB = h_unknownDouble + e.second;

            problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB);
            edgesAdded++;
        }
    }
    cout << "Edges added: " << edgesAdded << endl;

    // add all fit constraints
    set<int> addedEdges;
    for (auto &e : edges)
    {
        const float mask = h_mask[e.first];
        if (mask != 0.0f && addedEdges.count(e.first) == 0)
        {
            addedEdges.insert(e.first);
            const vec4f target = toVec(h_unknownFloat[e.first]);

            ceres::CostFunction* costFunction = FitTerm::Create(target, 1.0f);
            double4 *varStart = h_unknownDouble + e.first;

            problem.AddResidualBlock(costFunction, NULL, (double*)varStart);
            edgesAdded++;
        }
    }

    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = true;

    //faster methods
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s

    //slower methods
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
    //options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s

    //options.minimizer_type = ceres::LINE_SEARCH;

    //options.min_linear_solver_iterations = linearIterationMin;
    options.max_num_iterations = 100;
    options.function_tolerance = 0.01;
    options.gradient_tolerance = 1e-4 * options.function_tolerance;

    //options.min_lm_diagonal = 1.0f;
    //options.min_lm_diagonal = options.max_lm_diagonal;
    //options.max_lm_diagonal = 10000000.0;

    //problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    //cout << "Cost*2 start: " << cost << endl;

    double elapsedTime;
    {
        ml::Timer timer;
        try {
            Solve(options, &problem, &summary);
        }
        catch (exception ex)
        {
            cout << "exception: " << ex.what() << endl;
        }
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

    for (int i = 0; i < pixelCount; i++)
    {
        h_unknownFloat[i].x = (float)h_unknownDouble[i].x;
        h_unknownFloat[i].y = (float)h_unknownDouble[i].y;
        h_unknownFloat[i].z = (float)h_unknownDouble[i].z;
        h_unknownFloat[i].w = (float)h_unknownDouble[i].w;
    }

    cout << "Final time: " << (float)(summary.total_time_in_seconds * 1000.0) << "ms" << endl;
}

#endif
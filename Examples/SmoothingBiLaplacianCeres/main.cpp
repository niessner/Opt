// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "main.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const int kStride = 4;

struct EdgeConstraint {
    typedef DynamicAutoDiffCostFunction<EdgeConstraint, kStride> EdgeCostFunction;

    EdgeConstraint(int _pixel0, int _pixel1, float _weight)
        : pixel0(_pixel0), pixel1(_pixel1), weight(_weight) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        residuals[0] = (parameters[0][0] - parameters[1][0]) * T(weight);
        return true;
    }

    static EdgeCostFunction* Create(int pixel0, int pixel1, float weight,
        vector<double>& values,
        vector<double*>& parameterBlocks) {
        auto constraint = new EdgeConstraint(pixel0, pixel1, weight);
        auto costFunction = new EdgeCostFunction(constraint);
        
        // Add all the parameter blocks that affect this constraint.
        parameterBlocks.clear();
        
        parameterBlocks.push_back(&(values[pixel0]));
        costFunction->AddParameterBlock(1);

        parameterBlocks.push_back(&(values[pixel1]));
        costFunction->AddParameterBlock(1);

        costFunction->SetNumResiduals(1);
        return costFunction;
    }

private:
    const int pixel0;
    const int pixel1;
    const float weight;
};

struct ReconstructionConstraint {
    typedef DynamicAutoDiffCostFunction<ReconstructionConstraint, kStride> ReconstructionCostFunction;

    ReconstructionConstraint(int _pixel, float _value, float _weight)
        : pixel(_pixel), value(_value), weight(_weight) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        residuals[0] = (parameters[0][0] - T(value)) * T(weight);
        return true;
    }

    static ReconstructionCostFunction* Create(int pixel, float value, float weight,
        vector<double>& values,
        vector<double*>& parameterBlocks) {
        auto constraint = new ReconstructionConstraint(pixel, value, weight);
        auto costFunction = new ReconstructionCostFunction(constraint);

        // Add all the parameter blocks that affect this constraint.
        parameterBlocks.clear();

        parameterBlocks.push_back(&(values[pixel]));
        costFunction->AddParameterBlock(1);

        costFunction->SetNumResiduals(1);
        return costFunction;
    }

private:
    const int pixel;
    const float value;
    const float weight;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    auto image = LodePNG::load("input.png");

    Grid2<float> grid(image.getDimX(), image.getDimY());
    for (auto &p : image)
    {
        grid(p.x, p.y) = p.value.r / 255.0f;
    }

    const int width = (int)grid.getDimX();
    const int height = (int)grid.getDimY();
    const float edgeWeight = 1.0f;
    const float reconWeight = 0.1f;

    auto getPixelIndex = [=](int x, int y)
    {
        return (y * width + x);
    };

    vector<vec2i> edges;
    for (int x = 0; x < width - 1; x++)
    {
        for (int y = 0; y < height - 1; y++)
        {
            edges.push_back(vec2i(getPixelIndex(x, y), getPixelIndex(x + 1, y)));
            edges.push_back(vec2i(getPixelIndex(x, y), getPixelIndex(x, y + 1)));
        }
    }

    for (int x = 0; x < width - 1; x++)
        edges.push_back(vec2i(getPixelIndex(x, height - 1), getPixelIndex(x + 1, height - 1)));

    for (int y = 0; y < height - 1; y++)
        edges.push_back(vec2i(getPixelIndex(width - 1, y), getPixelIndex(width - 1, y + 1)));

    vector<double> variables(width * height);
    for (auto &v : variables)
        v = 0.0;

    Problem problem;

    // add all edge constraints
    for (auto e : edges)
    {
        vector<double*> parameterBlocks;
        auto costFunction = EdgeConstraint::Create(e.x, e.y, edgeWeight, variables, parameterBlocks);
        problem.AddResidualBlock(costFunction, NULL, parameterBlocks);
    }

    // add all reconstruction constraints
    for (const auto &v : grid)
    {
        vector<double*> parameterBlocks;
        auto costFunction = ReconstructionConstraint::Create(getPixelIndex(v.x, v.y), v.value, reconWeight, variables, parameterBlocks);
        problem.AddResidualBlock(costFunction, NULL, parameterBlocks);
    }

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = true;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::LinearSolverType::CGNR;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
    
    options.min_lm_diagonal = options.max_lm_diagonal;
    //options.max_lm_diagonal = 10000000.0;
    
    cout << "Solving..." << endl;

    double elapsedTime;
    {
        ml::Timer timer;
        Solve(options, &problem, &summary);
        elapsedTime = timer.getElapsedTimeMS();
    }

    cout << "Done, " << elapsedTime << "ms" << endl;

    for (auto &p : image)
    {
        unsigned char v = util::boundToByte(variables[getPixelIndex(p.x, p.y)] * 255.0f);
        p.value.r = v;
        p.value.g = v;
        p.value.b = v;
    }
    LodePNG::save(image, "output.png");

    cout << summary.FullReport() << endl;
    
    cout << "Final values:" << endl;
    for (const auto &v : grid)
    {
        //cout << "(" << v.x << "," << v.y << ") = " << variables[getPixelIndex(v.x, v.y)] << endl;
    }
    return 0;
}

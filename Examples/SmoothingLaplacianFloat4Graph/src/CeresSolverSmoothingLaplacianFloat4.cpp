
#include "main.h"

#ifdef USE_CERES

#include "CeresSolverSmoothingLaplacianFloat4.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

const int kStride = 4;

struct RegConstraint {
    typedef DynamicAutoDiffCostFunction<RegConstraint, kStride> RegCostFunction;

    RegConstraint(int _var0, int _var1, float _weight)
        : var0(_var0), var1(_var1), weight(_weight) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        residuals[0] = (parameters[0][0] - parameters[1][0]) * T(weight);
        return true;
    }

    static RegCostFunction* Create(int var0, int var1, float weight,
        vector<double>& values,
        vector<double*>& parameterBlocks) {
        auto constraint = new RegConstraint(var0, var1, weight);
        auto costFunction = new RegCostFunction(constraint);

        // Add all the parameter blocks that affect this constraint.
        parameterBlocks.clear();

        parameterBlocks.push_back(&(values[var0]));
        costFunction->AddParameterBlock(1);

        parameterBlocks.push_back(&(values[var1]));
        costFunction->AddParameterBlock(1);

        costFunction->SetNumResiduals(1);
        return costFunction;
    }

private:
    const int var0;
    const int var1;
    const float weight;
};

struct FitConstraint {
    typedef DynamicAutoDiffCostFunction<FitConstraint, kStride> FitCostFunction;

    FitConstraint(int _pixel, float _value, float _weight)
        : pixel(_pixel), value(_value), weight(_weight) {}

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        residuals[0] = (parameters[0][0] - T(value)) * T(weight);
        return true;
    }

    static FitCostFunction* Create(int pixel, float value, float weight,
        vector<double>& values,
        vector<double*>& parameterBlocks) {
        auto constraint = new FitConstraint(pixel, value, weight);
        auto costFunction = new FitCostFunction(constraint);

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

vector < vec2i > CeresSolverSmoothingLaplacianFloat4::makeImageEdges()
{
    auto getVarIndex = [=](int x, int y, int c)
    {
        return c * _width * _height + y * _width + x;
    };

    vector<vec2i> edges;
    for (int c = 0; c < 4; c++)
    {
        for (int x = 0; x < _width - 1; x++)
        {
            for (int y = 0; y < _height - 1; y++)
            {
                edges.push_back(vec2i(getVarIndex(x, y, c), getVarIndex(x + 1, y, c)));
                edges.push_back(vec2i(getVarIndex(x, y, c), getVarIndex(x, y + 1, c)));
            }
        }

        for (int x = 0; x < _width - 1; x++)
            edges.push_back(vec2i(getVarIndex(x, _height - 1, c), getVarIndex(x + 1, _height - 1, c)));

        for (int y = 0; y < _height - 1; y++)
            edges.push_back(vec2i(getVarIndex(_width - 1, y, c), getVarIndex(_width - 1, y + 1, c)));
    }
    return edges;
}

void CeresSolverSmoothingLaplacianFloat4::solve(const ColorImageR32G32B32A32 &image, float weightFit, float weightReg, ColorImageR32G32B32A32 &result)
{
    google::InitGoogleLogging("debug.exe");

    _width = image.getWidth();
    _height = image.getHeight();

    weightFit = sqrt(weightFit);
    weightReg = sqrt(weightReg);

    auto getVarIndex = [=](int x, int y, int c)
    {
        return c * _width * _height + y * _width + x;
    };

    /*local w_fit_rt, w_reg_rt = math.sqrt(w_fit), math.sqrt(w_reg)
        local cost = ad.sumsquared(w_fit_rt*(X(0, 0) - A(0, 0)),
        w_reg_rt*(X(G.v0) - X(G.v1)),
        w_reg_rt*(X(G.v1) - X(G.v0)))
        return adP:Cost(cost)*/
    
    cout << "Setting up problem..." << endl;

    vector<vec2i> edges = makeImageEdges();
    
    vector<double> variables(_width * _height * 4);
    for (auto &v : variables)
        v = 0.0;

    Problem problem;

    // add all reg constraints
    for (auto e : edges)
    {
        vector<double*> parameterBlocks;
        auto costFunction = RegConstraint::Create(e.x, e.y, weightReg, variables, parameterBlocks);
        problem.AddResidualBlock(costFunction, NULL, parameterBlocks);
    }

    // add all fit constraints
    for (int c = 0; c < 4; c++)
    {
        for (const auto &p : image)
        {
            vector<double*> parameterBlocks;
            auto costFunction = FitConstraint::Create(getVarIndex(p.x, p.y, c), p.value[c], weightFit, variables, parameterBlocks);
            problem.AddResidualBlock(costFunction, NULL, parameterBlocks);
        }
    }

    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    options.minimizer_progress_to_stdout = true;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::LinearSolverType::CGNR;
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;

    //options.min_lm_diagonal = options.max_lm_diagonal;
    //options.max_lm_diagonal = 10000000.0;

    double elapsedTime;
    {
        ml::Timer timer;
        Solve(options, &problem, &summary);
        elapsedTime = timer.getElapsedTimeMS();
    }

    cout << "Done, " << elapsedTime << "ms" << endl;

    cout << summary.FullReport() << endl;

    for (int c = 0; c < 4; c++)
    {
        for (auto &p : result)
        {
            p.value[c] = variables[getVarIndex(p.x, p.y, c)];
        }
    }
}

#endif
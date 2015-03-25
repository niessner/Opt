
#include "main.h"

TestExample TestFramework::makeRandomQuadratic(int count)
{
    TestExample result("quadratic1D", "quadratic.t", count);
    
    //
    // image order: x, a, b, c
    //
    result.images.resize(4);
    for (auto &image : result.images)
        image.allocate(count, 1);

    for (int i = 0; i < count; i++)
    {
        result.images[1](i, 0) = i * 0.1 + 1.0;
        result.images[2](i, 0) = i * 0.1 + 2.0;
        result.images[3](i, 0) = i * 0.1 + 3.0;
    }

    //
    // residual_i = ( (ax^2 + bx + c)^2 = 0)
    //

    result.costFunction = [=](const double *variables)
    {
        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            const double x = variables[i];
            const double a = result.images[1](i, 0);
            const double b = result.images[2](i, 0);
            const double c = result.images[3](i, 0);
            const double v = a * x * x + b * x + c;
            sum += v * v;
        }
        return sum;
    };

    double minimumSum = 0.0;
    for (int i = 0; i < count; i++)
    {
        const double a = result.images[1](i, 0);
        const double b = result.images[2](i, 0);
        const double c = result.images[3](i, 0);
        const double x = -b / (2.0 * a);
        const double v = a * x * x + b * x + c;
        minimumSum += v * v;
    }
    result.minimumCost = minimumSum;

    return result;
}

void TestFramework::runAllTests()
{
    optimizerState = Opt_NewState();

    TestExample example = makeRandomQuadratic(5);
    TestMethod method = TestMethod("gradientdescent","no-params");

    for (auto &image : example.images)
        image.bind(optimizerState);

    runTest(method, example);
}

void TestFramework::runTest(const TestMethod &method, const TestExample &example)
{
    cout << "Running test: " << example.exampleName << " using " << method.optimizerName << endl;

    uint64_t dims[] = { example.variableDimX, example.variableDimY };

    Problem * prob = Opt_ProblemDefine(optimizerState, example.terraCodeFilename.c_str(), method.optimizerName.c_str(), NULL);

    if (!prob)
    {
        cout << "Opt_ProblemDefine failed" << endl;
        cin.get();
        return;
    }

    Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims);

    vector<ImageBinding *> imageBindings;
    for (const auto &image : example.images)
        imageBindings.push_back(image.terraBinding);

    Opt_ProblemSolve(optimizerState, plan, imageBindings.data(), NULL);

    cout << "expected cost: " << example.minimumCost << endl;
}

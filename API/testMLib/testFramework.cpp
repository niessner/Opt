
#include "main.h"

TestExample TestFramework::makeRandomQuadratic(int count)
{
    TestExample result("quadratic1D", "quadratic.t", count, 1);
    
    //
    // image order: x, a, b, c
    //
    result.images.resize(4);
    for (auto &image : result.images)
        image.allocate(count, 1);

    for (int i = 0; i < count; i++)
    {
        result.images[1](i, 0) = i * 0.1f + 1.0f;
        result.images[2](i, 0) = i * 0.1f + 2.0f;
        result.images[3](i, 0) = i * 0.1f + 3.0f;
    }

    //
    // residual_i = ( (ax^2 + bx + c)^2 = 0)
    //

    result.costFunction = [=](const OptImage &variables)
    {
        float sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            const float x = variables(i, 0);
            const float a = result.images[1](i, 0);
            const float b = result.images[2](i, 0);
            const float c = result.images[3](i, 0);
            const float v = a * x * x + b * x + c;
            sum += v;
        }
        return sum;
    };

    float minimumSum = 0.0f;
    for (int i = 0; i < count; i++)
    {
        const float a = result.images[1](i, 0);
        const float b = result.images[2](i, 0);
        const float c = result.images[3](i, 0);
        const float x = -b / (2.0f * a);
        const float v = a * x * x + b * x + c;
        minimumSum += v;
    }
    result.minimumCost = minimumSum;

    return result;
}

void TestFramework::runAllTests()
{
    optimizerState = Opt_NewState();
    if (optimizerState == nullptr)
    {
        cerr << "Opt_NewState failed" << endl;
        return;
    }

    //TestExample example = makeRandomQuadratic(5);
    TestExample example = makeImageSmoothing("smoothingExampleB.png", 0.1f);
    //TestExample example = makeMeshSmoothing("smoothingExampleB.png", 0.1f);

    vector<TestMethod> methods;

    //
    // CPU methods
    //
    //methods.push_back(TestMethod("gradientDescentCPU","no-params"));
    //methods.push_back(TestMethod("conjugateGradientCPU", "no-params"));
    //methods.push_back(TestMethod("linearizedConjugateGradientCPU", "no-params"));
    //methods.push_back(TestMethod("lbfgsCPU", "no-params"));
    //methods.push_back(TestMethod("vlbfgsCPU", "no-params"));
    //methods.push_back(TestMethod("bidirectionalVLBFGSCPU", "no-params"));

    //
    // GPU methods
    //
	//methods.push_back(TestMethod("vlbfgsGPU", "no-params"));
	//methods.push_back(TestMethod("adaDeltaGPU", "no-params"));
    //methods.push_back(TestMethod("gradientDescentGPU", "no-params"));
	//methods.push_back(TestMethod("gaussNewtonGPU", "no-params"));
	methods.push_back(TestMethod("gaussNewtonBlockGPU", "no-params"));

    for (auto &method : methods)
        runTest(method, example);
}

void TestFramework::runTest(const TestMethod &method, TestExample &example)
{
    example.images[0].clear(0.0f);

    cout << "Running test: " << example.exampleName << " using " << method.optimizerName << endl;

    cout << "start cost: " << example.costFunction(example.images[0]) << endl;

    uint64_t dims[] = { example.variableDimX, example.variableDimY };

    Problem * prob = Opt_ProblemDefine(optimizerState, example.terraCodeFilename.c_str(), method.optimizerName.c_str(), NULL);

    if (!prob)
    {
        cout << "Opt_ProblemDefine failed" << endl;
        #ifdef _WIN32
        cin.get();
        #endif
        return;
    }
    
    vector<void*> imagesCPU;
    vector<void*> imagesGPU;
    vector<uint64_t> stride;
    vector<uint64_t> elemsize;

    for (const auto &image : example.images)
    {
        image.syncCPUToGPU();
        imagesCPU.push_back((void*)image.DataCPU());
        imagesGPU.push_back((void*)image.DataGPU());
        stride.push_back(image.dimX * sizeof(float));
        elemsize.push_back(sizeof(float));
    }

    vector<int64_t*> adjacencyOffsetsCPU;
    vector<int64_t*> adjacencyListsXCPU;
    vector<int64_t*> adjacencyListsYCPU;
    vector<void*> edgeValuesCPU;

    for (auto &graph : example.graphs)
    {
        graph.finalize();
        adjacencyOffsetsCPU.push_back((int64_t *)graph.adjacencyOffsetsCPU.data());
        adjacencyListsXCPU.push_back((int64_t *)graph.adjacencyListsXCPU.data());
        adjacencyListsYCPU.push_back((int64_t *)graph.adjacencyListsYCPU.data());
        edgeValuesCPU.push_back((void*)graph.edgeValuesCPU.data());
    }
    Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims, elemsize.data(), stride.data(), adjacencyOffsetsCPU.data(), adjacencyListsXCPU.data(), adjacencyListsYCPU.data());
    
    //Plan * plan = Opt_ProblemPlan(optimizerState, prob, dims,
    //    elemsize.data(), stride.data(),
    //    adjacencyOffsetsCPU.data(), adjacencyListsCPU.data());

    if (!plan)
    {
        cout << "Opt_ProblemPlan failed" << endl;
        #ifdef _WIN32
        cin.get();
        #endif
        return;
    }

    bool isGPU = ml::util::endsWith(method.optimizerName, "GPU");



    if (isGPU)
    {
        Opt_ProblemSolve(optimizerState, plan, imagesGPU.data(), edgeValuesCPU.data(), NULL);
        for (const auto &image : example.images)
            image.syncGPUToCPU();
    }
    else
    {
        //Opt_ProbledemSolve(optimizerState, plan, imagesCPU.data(), edgeValuesCPU.data());
        Opt_ProblemSolve(optimizerState, plan, imagesCPU.data(), edgeValuesCPU.data(), NULL);
    }

    //cout << "x(0, 0) = " << example.images[0](0, 0) << endl;
    //cout << "x(1, 0) = " << example.images[0](5, 0) << endl;
    //cout << "x(0, 1) = " << example.images[0](0, 5) << endl;

    // TODO: this is not always accurate, in cases where costFunction does not exactly match the cost function in the terra file.  This should just call terra's cost function.
    cout << "optimized cost: " << example.costFunction(example.images[0]) << endl;

    cout << "expected cost: " << example.minimumCost << endl;

    cout << "max delta: " << OptImage::maxDelta(example.images[0], example.minimumValues) << endl;
    cout << "avg delta: " << OptImage::avgDelta(example.images[0], example.minimumValues) << endl;
}

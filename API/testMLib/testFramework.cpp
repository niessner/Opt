
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
    //TestExample example = makeMeshSmoothing("smoothingExampleE.png", 0.1f);
	TestExample example = makeImageSmoothing("smoothingExampleB.png", "imageSmoothingAD.t", 0.1f);	

    vector<TestMethod> methods;

    //
    // CPU methods
    //
    //methods.push_back(TestMethod("gradientDescentCPU","no-params"));

    // 
    // GPU methods
    //
    //methods.push_back(TestMethod("gradientDescentGPU", "no-params"));
	//methods.push_back(TestMethod("gaussNewtonGPU", "no-params"));
	methods.push_back(TestMethod("gaussNewtonBlockGPU", "no-params"));

    for (auto &method : methods)
        runTest(method, example);
}

void TestFramework::runTest(const TestMethod &method, TestExample &example)
{
    example.images[0].clear(0.0f);

    cerr << "Running test: " << example.exampleName << " using " << method.optimizerName << endl;

    cerr << "start cost: " << example.costFunction(example.images[0]) << endl;

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
    
    if (!plan)
    {
        cerr << "Opt_ProblemPlan failed" << endl;
        #ifdef _WIN32
        cin.get();
        #endif
        return;
    }

    bool isGPU = ml::util::endsWith(method.optimizerName, "GPU");

    int a = 0;
    void * list[] = { &a };
    cerr << "Problem solve" << endl;
    Opt_ProblemSolve(optimizerState, plan, isGPU ? imagesGPU.data() : imagesCPU.data(), edgeValuesCPU.data(), list, NULL);
    if (isGPU) {
        for (const auto &image : example.images)
            image.syncGPUToCPU();
    }

    //cout << "x(0, 0) = " << example.images[0](0, 0) << endl;
    //cout << "x(1, 0) = " << example.images[0](5, 0) << endl;
    //cout << "x(0, 1) = " << example.images[0](0, 5) << endl;

    // TODO: this is not always accurate, in cases where costFunction does not exactly match the cost function in the terra file.  This should just call terra's cost function.
    cerr << "optimized cost: " << example.costFunction(example.images[0]) << endl;

    cerr << "expected cost: " << example.minimumCost << endl;

    cerr << "max delta: " << OptImage::maxDelta(example.images[0], example.minimumValues) << endl;
    cerr << "avg delta: " << OptImage::avgDelta(example.images[0], example.minimumValues) << endl;
}


struct TestMethod
{
    TestMethod(const string &_optimizerName, const string &_optimizerParameters)
    {
        optimizerName = _optimizerName;
        optimizerParameters = _optimizerParameters;
    }

    string optimizerName;
    string optimizerParameters;
};

struct TestImage
{
    TestImage()
    {
        terraBindingCPU = nullptr;
        terraBindingGPU = nullptr;
        dataGPU = nullptr;
    }
    TestImage(int _dimX, int _dimY)
    {
        allocate(_dimX, _dimY);
    }

    void allocate(int _dimX, int _dimY)
    {
        dimX = _dimX;
        dimY = _dimY;
        dataCPU.resize(dimX * dimY);
        cudaMalloc(&dataGPU, sizeof(double) * dimX * dimY);
    }
    void syncCPUToGPU() const
    {
        cudaMemcpy(dataGPU, (void *)dataCPU.data(), sizeof(double) * dimX * dimY, cudaMemcpyHostToDevice);
    }
    void bind(OptState *optimizerState)
    {
        terraBindingCPU = Opt_ImageBind(optimizerState, dataCPU.data(), sizeof(double), dimX * sizeof(double));
        terraBindingGPU = Opt_ImageBind(optimizerState, dataGPU, sizeof(double), dimX * sizeof(double));
    }
    double& operator()(int x, int y)
    {
        return dataCPU[y * dimX + x];
    }
    double operator()(int x, int y) const
    {
        return dataCPU[y * dimX + x];
    }

    ImageBinding *terraBindingCPU;
    ImageBinding *terraBindingGPU;
    vector<double> dataCPU;
    void *dataGPU;
    int dimX, dimY;
};

struct TestExample
{
    TestExample(const string &_exampleName, const string &_terraCodeFilename, size_t _variableCount)
    {
        exampleName = _exampleName;
        terraCodeFilename = _terraCodeFilename;
        variableDimX = _variableCount;
        variableDimY = _variableCount;
    }

    size_t variableDimX, variableDimY;
    string exampleName;
    string terraCodeFilename;

    vector<TestImage> images;

    double minimumCost;

    function<double(const double *variables)> costFunction;
};

class TestFramework
{
public:
    void runAllTests();

private:
    void runTest(const TestMethod &method, const TestExample &example);

    TestExample makeRandomQuadratic(int count);

    vector<TestMethod> methods;
    vector<TestExample> examples;

    OptState *optimizerState;
};
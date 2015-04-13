
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
		cudaMalloc(&dataGPU, sizeof(float) * dimX * dimY);
    }
    void syncCPUToGPU() const
    {
		cudaMemcpy(dataGPU, (void *)dataCPU.data(), sizeof(float) * dimX * dimY, cudaMemcpyHostToDevice);
    }
    void syncGPUToCPU() const
    {
        cudaMemcpy((void *)dataCPU.data(), dataGPU, sizeof(float) * dimX * dimY, cudaMemcpyDeviceToHost);
    }
    void bind(OptState *optimizerState)
    {
        terraBindingCPU = Opt_ImageBind(optimizerState, dataCPU.data(), sizeof(float), dimX * sizeof(float));
		terraBindingGPU = Opt_ImageBind(optimizerState, dataGPU, sizeof(float), dimX * sizeof(float));
    }
    float& operator()(int x, int y)
    {
        return dataCPU[y * dimX + x];
    }
	float operator()(int x, int y) const
    {
        return dataCPU[y * dimX + x];
    }
    float& operator()(size_t x, size_t y)
    {
        return dataCPU[y * dimX + x];
    }
    float operator()(size_t x, size_t y) const
    {
        return dataCPU[y * dimX + x];
    }
    static double maxDelta(const TestImage &a, const TestImage &b)
    {
        double delta = 0.0;
        for (int x = 0; x < a.dimX; x++)
            for (int y = 0; y < a.dimY; y++)
                delta = std::max(delta, fabs((double)a(x, y) - (double)b(x, y)));
        return delta;
    }

    ImageBinding *terraBindingCPU;
    ImageBinding *terraBindingGPU;
    vector<float> dataCPU;
    void *dataGPU;
    int dimX, dimY;
};

struct TestExample
{
    TestExample(const string &_exampleName, const string &_terraCodeFilename, size_t _variableDimX, size_t _variableDimY)
    {
        exampleName = _exampleName;
        terraCodeFilename = _terraCodeFilename;
        variableDimX = _variableDimX;
        variableDimY = _variableDimY;
    }

    size_t variableDimX, variableDimY;
    string exampleName;
    string terraCodeFilename;

    vector<TestImage> images;

    float minimumCost;
    TestImage minimumValues;

    function<float(const TestImage &x)> costFunction;
};

class TestFramework
{
public:
    void runAllTests();

private:
    void runTest(const TestMethod &method, const TestExample &example);

    TestExample makeRandomQuadratic(int count);
    TestExample makeImageSmoothing(const string &imageFilename, float w);

    vector<TestMethod> methods;
    vector<TestExample> examples;

    OptState *optimizerState;
};

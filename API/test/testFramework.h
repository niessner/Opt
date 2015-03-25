
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
        terraBinding = NULL;
    }
    TestImage(int _dimX, int _dimY)
    {
        allocate(_dimX, _dimY);
    }

    void allocate(int _dimX, int _dimY)
    {
        dimX = _dimX;
        dimY = _dimY;
        data.resize(dimX * dimY);
    }
    void bind(OptState *optimizerState)
    {
        terraBinding = Opt_ImageBind(optimizerState, data.data(), sizeof(double), dimX * sizeof(double));
    }
    double& operator()(int x, int y)
    {
        return data[y * dimX + x];
    }
    double operator()(int x, int y) const
    {
        return data[y * dimX + x];
    }

    ImageBinding *terraBinding;
    vector<double> data;
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
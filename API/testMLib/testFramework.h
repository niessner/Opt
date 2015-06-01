

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

struct TestExample
{
	TestExample(const string &_exampleName, const string &_terraCodeFilename, uint32_t _variableDimX, uint32_t _variableDimY)
    {
        exampleName = _exampleName;
        terraCodeFilename = _terraCodeFilename;
        variableDimX = _variableDimX;
        variableDimY = _variableDimY;
    }

    uint32_t variableDimX, variableDimY;
    string exampleName;
    string terraCodeFilename;

    vector<OptImagef> images;
    vector<OptGraphf> graphs;

    float minimumCost;
    OptImagef minimumValues;

    function<float(const OptImagef &x)> costFunction;
};

class TestFramework
{
public:
    void runAllTests(int argc, char ** argv);

private:
    void runTest(const TestMethod &method, TestExample &example);

    TestExample makeRandomQuadratic(int count);
    TestExample makeImageSmoothing(const string &imageFilename, const string & terraCodeFilename, float w);
    TestExample makeMeshSmoothing(const string &imageFilename, const string & terraCodeFilename, float w);

    vector<TestMethod> methods;
    vector<TestExample> examples;

    OptState *optimizerState;
};

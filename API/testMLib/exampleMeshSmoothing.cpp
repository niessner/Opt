
#include "main.h"

const bool groundTruth = false;

TestExample TestFramework::makeMeshSmoothing(const string &imageFilename,const string &terraCodeFilename, float w)
{
    const Bitmap bmp = LodePNG::load(imageFilename);
    const int dimX = bmp.getWidth();
    const int dimY = bmp.getHeight();

    const int pixelCount = bmp.size();

    OptGraphf graph(pixelCount);

    auto getVariable = [=](int x, int y)
    {
        return (y * dimX + x);
    };

    for (const auto &p : bmp)
    {
        graph.nodes[getVariable((int)p.x, (int)p.y)].x = (int)p.x;
        graph.nodes[getVariable((int)p.x, (int)p.y)].y = (int)p.y;
    }

    auto getNode = [&](int x, int y) -> OptGraphf::Node&
    {
        return graph.nodes[getVariable(x, y)];
    };

    auto addEdge = [&](int xStart, int yStart, int xOffset, int yOffset)
    {
        int x = xStart + xOffset;
        int y = yStart + yOffset;
        if (x >= 0 && x < dimX && y >= 0 && y < dimY)
        {
            auto &nodeStart = getNode(xStart, yStart);
            auto &nodeEnd = getNode(x, y);
            nodeStart.edges.push_back(OptGraphf::Edge(nodeEnd.index, 1.0f));
        }
    };

    for (const auto &p : bmp)
    {
        addEdge((int)p.x, (int)p.y, -1, 0);
        addEdge((int)p.x, (int)p.y, 1, 0);
        addEdge((int)p.x, (int)p.y, 0, -1);
        addEdge((int)p.x, (int)p.y, 0, 1);
    }

    MathVector<float> x(pixelCount);
    if (groundTruth)
    {
        SparseMatrixf L(pixelCount, pixelCount);
        MathVector<float> targetValues(pixelCount);

        for (const auto &n : graph.nodes)
        {
            L((size_t)n.index, (size_t)n.index) = (float)n.edges.size();

            for (const auto &e : n.edges)
            {
                L((size_t)n.index, (size_t)e.end) = -1.0;
            }
        }

        
        for (const auto &p : bmp)
            targetValues[getVariable((int)p.x, (int)p.y)] = p.value.r;

        SparseMatrixf W = SparseMatrixf::identity(pixelCount) * w;

        SparseMatrixf A = L.transpose() * L + W;
        MathVector<float> b = W * targetValues;

        LinearSolverConjugateGradient<float> solver;
        x = solver.solve(A, b);
    }

    TestExample result("meshSmoothing", terraCodeFilename, bmp.getWidth(), bmp.getHeight());

    result.graphs.push_back(graph);

    Bitmap testImage = bmp;
    for (const auto &p : bmp)
        testImage(p.x, p.y) = vec4uc(util::boundToByte(x[getVariable((int)p.x, (int)p.y)]));

    LodePNG::save(testImage, "smoothingOutputLinearSolve.png");

    result.costFunction = [=](const OptImage &variables)
    {
        float sum = 0.0;

        //
        // Laplacian cost
        //
        for (const auto &n : graph.nodes)
        {
            const float x = variables(n.x, n.y);

            float nsum = 0.0f;
            for (const auto &e : n.edges)
                nsum += variables(graph.nodes[e.end].x, graph.nodes[e.end].y) * e.value;

            const float laplacianCost = n.edges.size() * x - nsum;

            sum += laplacianCost * laplacianCost;
        }

        //
        // Reconstruction cost
        //
        for (const auto &p : bmp)
        {
            const float x = variables(p.x, p.y);
            const float reconstructionCost = x - p.value.r;

            sum += w * (reconstructionCost * reconstructionCost);
        }

        return sum;
    };

    result.images.resize(2);
    result.images[0].allocate(bmp.getWidth(), bmp.getHeight());
    result.images[1].allocate(bmp.getWidth(), bmp.getHeight());
    result.minimumValues = result.images[0];

    for (const auto &p : bmp)
    {
        result.images[0](p.x, p.y) = 0.0;
        result.images[1](p.x, p.y) = p.value.r;
        result.minimumValues(p.x, p.y) = x[getVariable((int)p.x, (int)p.y)];
    }

    result.minimumCost = result.costFunction(result.minimumValues);

    return result;
}

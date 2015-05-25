
#include "main.h"

const bool groundTruth = false;

TestExample TestFramework::makeImageSmoothing(const string &imageFilename, float w)
{
    //
    // terms:
    // smoothness: 4 * x_i - (neighbors) = 0
    // reconstruction: x_i = c_i
    // 
    // final energy function:
    // E(x) = sum_i( (4 * x_i - (neighbors) ) ^2 ) + sum_i( w * (x_i - target_i)^2 )
    //
    // minimized with A = L^T L + I * w, b = I * w * target

    const Bitmap bmp = LodePNG::load(imageFilename);
    const int dimX = bmp.getWidth();
    const int dimY = bmp.getHeight();

    const size_t pixelCount = bmp.size();

    auto getVariable = [=](size_t x, size_t y)
    {
        return (size_t)(y * dimX + x);
    };

    auto isBorder = [=](size_t x, size_t y)
    {
        return (x == 0 || y == 0 || x == dimX - 1 || y == dimY - 1);
    };

    MathVector<float> x(pixelCount);
    if (groundTruth)
    {
        SparseMatrixf L(pixelCount, pixelCount);
        for (const auto &p : bmp)
        {
            if (isBorder(p.x, p.y))
                continue;

            size_t row = getVariable(p.x, p.y);
            L(row, row) = 4.0;
            L(row, getVariable(p.x - 1, p.y + 0)) = -1.0;
            L(row, getVariable(p.x + 1, p.y + 0)) = -1.0;
            L(row, getVariable(p.x + 0, p.y - 1)) = -1.0;
            L(row, getVariable(p.x + 0, p.y + 1)) = -1.0;
        }

        MathVector<float> targetValues(pixelCount);
        for (const auto &p : bmp)
            targetValues[getVariable(p.x, p.y)] = p.value.r;

        SparseMatrixf W = SparseMatrixf::identity(pixelCount) * w;

        SparseMatrixf A = L.transpose() * L + W;
        MathVector<float> b = W * targetValues;

        LinearSolverConjugateGradient<float> solver;
        x = solver.solve(A, b);
    }

    Bitmap testImage = bmp;
    for (const auto &p : bmp)
        testImage(p.x, p.y) = vec4uc(util::boundToByte(x[getVariable(p.x, p.y)]));

    LodePNG::save(testImage, "smoothingOutputLinearSolve.png");
    
    TestExample result("imageSmoothing", "imageSmoothingCombined.t", bmp.getWidth(), bmp.getHeight());
	//TestExample result("imageSmoothing", "imageSmoothingAD.t", bmp.getWidth(), bmp.getHeight());

    result.costFunction = [=](const OptImage &variables)
    {
        //(4 * x_i - (neighbors) ) ^2 ) + sum_i( w * (x_i - target_i)^2
        float sum = 0.0;

        //
        // Laplacian cost
        //
        for (const auto &p : bmp)
        {
            if (isBorder(p.x, p.y))
                continue;

            const float x = variables(p.x, p.y);

            const float n0 = variables(p.x - 1, p.y);
            const float n1 = variables(p.x + 1, p.y);
            const float n2 = variables(p.x, p.y - 1);
            const float n3 = variables(p.x, p.y + 1);

            const float laplacianCost = 4 * x - (n0 + n1 + n2 + n3);

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

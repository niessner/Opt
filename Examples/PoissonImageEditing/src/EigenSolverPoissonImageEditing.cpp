#pragma once

#include "main.h"
#include <Eigen33b1/Eigen>
#include <Eigen33b1/IterativeLinearSolvers>
#include <cuda_runtime.h>

#define LEAST_SQ_CONJ_GRADIENT 0
#define SPARSE_QR 1

#define SOLVER LEAST_SQ_CONJ_GRADIENT

typedef Eigen::Triplet<float> Tripf;
typedef Eigen::SparseMatrix<float> SpMatrixf;
#include "EigenSolverPoissonImageEditing.h"
#include <Eigen33b1/OrderingMethods>

#if SOLVER == LEAST_SQ_CONJ_GRADIENT
typedef Eigen::LeastSquaresConjugateGradient<SpMatrixf > AxEqBSolver;
#elif SOLVER == SPARSE_QR
typedef Eigen::SparseQR<SpMatrixf, Eigen::COLAMDOrdering<int> > AxEqBSolver;
#endif

struct vec2iHash {
    size_t operator()(const vec2i& v) const {
        return std::hash < int > {}(v.x) ^ std::hash < int > {}(v.y);
    }
};


float4 sampleImage(float4* image, vec2i p, int W) {
    return image[p.y*W + p.x];
}

void setPixel(float4* image, vec2i p, int W, float r, float g, float b) {
    image[p.y*W + p.x].x = r;
    image[p.y*W + p.x].y = g;
    image[p.y*W + p.x].z = b;
}

void solveAxEqb(AxEqBSolver& solver, const Eigen::VectorXf& b, Eigen::VectorXf& x) {
    
#if SOLVER==LEAST_SQ_CONJ_GRADIENT
    x = solver.solveWithGuess(b, x);
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error() << std::endl;
#else
    x = solver.solve(b);
#endif
}

void EigenSolverPoissonImageEditing::solve(float4* h_unknownFloat, float4* h_target, float* h_mask, float weightFit, float weightReg)
{
    float weightFitSqrt = sqrt(weightFit);
    float weightRegSqrt = sqrt(weightReg);
    int numUnknowns = 0;
    std::unordered_map<vec2i, int, vec2iHash> pixelLocationsToIndex;
    std::vector<vec2i> pixelLocations;
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            if (h_mask[y*m_width + x] == 0.0f) {
                ++numUnknowns;
                vec2i p(x, y);
                pixelLocationsToIndex[p] =(int)pixelLocations.size();
                pixelLocations.push_back(p);
            }
        }
    }
    printf("# Unknowns: %d\n", numUnknowns);
    Eigen::VectorXf x_r(pixelLocations.size()), b_r(pixelLocations.size());
    Eigen::VectorXf x_g(pixelLocations.size()), b_g(pixelLocations.size());
    Eigen::VectorXf x_b(pixelLocations.size()), b_b(pixelLocations.size());

    b_r.setZero();
    b_g.setZero();
    b_b.setZero();

    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        float4 color = sampleImage(h_unknownFloat, p, m_width);
        x_r[i] = color.x;
        x_g[i] = color.y;
        x_b[i] = color.z;
    }
    SpMatrixf A(pixelLocations.size(), pixelLocations.size());
    A.setZero();
    printf("Constructing Matrix\n");
    std::vector<Tripf> entriesA;
    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        int row = i;
        int numInternalNeighbors = 0;
        float4 g_p = sampleImage(h_target, p, m_width);
        for (int off_y = -1; off_y <= 1; off_y += 2) {
            for (int off_x = -1; off_x <= 1; off_x += 2) {
                vec2i q(p.x + off_x, p.y + off_y);
                auto it = pixelLocationsToIndex.find(q);
                if (it != pixelLocationsToIndex.end()) {
                    ++numInternalNeighbors;
                    entriesA.push_back(Tripf(row, it->second, -1.0f));
                } else {
                    float4 f_star_q = sampleImage(h_unknownFloat, q, m_width);
                    b_r[i] += f_star_q.x;
                    b_g[i] += f_star_q.y;
                    b_b[i] += f_star_q.z;
                }
                float4 g_q = sampleImage(h_target, q, m_width);
                b_r[i] += (g_p.x - g_q.x);
                b_g[i] += (g_p.y - g_q.y);
                b_b[i] += (g_p.z - g_q.z);
            }
        }
        entriesA.push_back(Tripf(row, row, (float)numInternalNeighbors));
    }
    printf("Entries Set\n");
    A.setFromTriplets(entriesA.begin(), entriesA.end());
    printf("Sparse Matrix Constructed\n");
    A.makeCompressed();
    printf("Matrix Compressed\n");
    {
        float totalCost = 0.0f;
        
        float cost_r = (A*x_r - b_r).squaredNorm();
        float cost_g = (A*x_g - b_g).squaredNorm();
        float cost_b = (A*x_b - b_b).squaredNorm();
        totalCost = cost_r + cost_g + cost_b;
        printf("Initial Cost: %f : (%f, %f, %f)\n", totalCost, cost_r, cost_g, cost_b);

    }

    
    // fill A and b
    AxEqBSolver solver;
    printf("Solvers Initialized\n");
    solver.compute(A);
    printf("solver.compute(A)\n");
    solveAxEqb(solver, b_r, x_r);
    printf("Red solve done\n");
    solveAxEqb(solver, b_g, x_g);
    printf("Green solve done\n");
    solveAxEqb(solver, b_b, x_b);
    printf("Blue solve done\n");
    float totalCost = 0.0f;
 
    float cost_r = (A*x_r - b_r).squaredNorm(); 
    float cost_g = (A*x_g - b_g).squaredNorm();
    float cost_b = (A*x_b - b_b).squaredNorm();
    totalCost = cost_r + cost_g + cost_b;
    printf("Final Cost: %f : (%f, %f, %f)\n", totalCost, cost_r, cost_g, cost_b);

    for (int i = 0; i < pixelLocations.size(); ++i) {
        setPixel(h_unknownFloat, pixelLocations[i], m_width, x_r[i], x_g[i], x_b[i]);
    }
    

}
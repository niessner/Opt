#pragma once

#include "main.h"
#include <Eigen33b1/Eigen>
#include <Eigen33b1/IterativeLinearSolvers>
#include <cuda_runtime.h>
typedef Eigen::Triplet<float> Tripf;
#include "EigenSolverPoissonImageEditing.h"


float4 sampleImage(float4* image, vec2i p, int W) {

}

void EigenSolverPoissonImageEditing::solve(float4* h_unknownFloat, float4* h_target, float* h_mask, float weightFit, float weightReg)
{
    float weightFitSqrt = sqrt(weightFit);
    float weightRegSqrt = sqrt(weightReg);

    std::unordered_map<vec2i, int> pixelLocationsToIndex;
    std::vector<vec2i> pixelLocations;
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            if (h_mask[y*m_width + x] == 0.0f) {
                vec2i p(x, y);
                pixelLocationsToIndex[p] = pixelLocations.size();
                pixelLocations.push_back(p);
            }
        }
    }

    Eigen::VectorXd x_r(pixelLocations.size()), b_r(pixelLocations.size());
    Eigen::VectorXd x_g(pixelLocations.size()), b_g(pixelLocations.size());
    Eigen::VectorXd x_b(pixelLocations.size()), b_b(pixelLocations.size());

    b_r.setZero();
    b_g.setZero();
    b_b.setZero();

    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        float4 color = h_unknownFloat[p.y*m_width + p.x];
        x_r[i] = color.x;
        x_g[i] = color.y;
        x_b[i] = color.z;
    }
    Eigen::SparseMatrix<float> A(pixelLocations.size(), pixelLocations.size());

    std::vector<Tripf> entriesA;;
    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        int row = i;
        int numInternalNeighbors = 0;
        for (int off_y = -1; off_y <= 1; off_y += 2) {
            for (int off_x = -1; off_x <= 1; off_x += 2) {
                vec2i q(p.x + off_x, p.y + off_y);
                auto it = pixelLocationsToIndex.find(q);
                if (it != pixelLocationsToIndex.end()) {
                    ++numInternalNeighbors;
                    entriesA.push_back(Tripf(row, it->second, -1.0f));
                } else {
                    float4 color = h_unknownFloat[p.y*m_width + p.x];
                    b_r[i] += color.x;
                    b_g[i] += color.y;
                    b_b[i] += color.z;
                }
                b_r[i] += color.x;
                b_g[i] += color.y;
                b_b[i] += color.z;


            }
        }
        entriesA.push_back(Tripf(row, row, (float)numInternalNeighbors));
    }

    // fill A and b
    Eigen::LeastSquaresConjugateGradient<SparseMatrix<float> > lscg_r;
    lscg_r.compute(A);
    x = lscg_r.solve(b);
    std::cout << "#iterations:     " << lscg_r.iterations() << std::endl;
    std::cout << "estimated error: " << lscg_r.error() << std::endl;
    // update b, and solve again
    x = lscg_r.solve(b);

}
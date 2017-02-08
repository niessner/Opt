#pragma once

#include <assert.h>
#include <vector>
#include <memory>
#include <numeric>
#include "cudaUtil.h"

/** 
    Small wrapper class for connectivity.
    Does not allow for full expressivity Opt allows for;
    only currently supports graphs connecting 1D images.

    {m_indices[0][i], m_indices[1][i], ..., m_indices[m_indices.size()-1][i]}

    defines a single hyper-edge of the graph.
    
*/
class OptGraph {
public:
    OptGraph(std::vector<std::vector<unsigned int>> indices) : m_indices(indices) {}

    OptGraph(size_t edgeCount, size_t edgeSize) {
        m_indices.resize(edgeSize);
        for (size_t i = 0; i < edgeSize; ++i) {
            m_indices[i].resize(edgeCount);
        }
    }

    void fastSetIndex(size_t edgeIdx, size_t vertexIdx, unsigned index) {
        m_indices[vertexIdx][edgeIdx] = index;
    }

    void copyToGPU() {

    }

    int* edgeCountPtr() {
        return &m_edgeCount;
    }

private:
    // CPU storage
    std::vector<std::vector<unsigned int>> m_indices;
    std::vector<unsigned int*> m_gpuIndices;
    // Copy of m_gpuIndices.size() in int form for use by Opt
    int m_edgeCount = 0;
};
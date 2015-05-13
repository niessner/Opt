#ifndef OptGraph_h
#define OptGraph_h
#include <stdint.h>
#include "cuda_runtime.h"

//
// All edges in OptGraph are directed. Once finalize is called, all CPU and GPU data is allocated, and
// the graph values can be chagned by the adjacency is immutable.
//
template<class EdgeValueType>
struct OptGraph
{
    struct Edge
    {
        Edge() {}
        Edge(int _end, const EdgeValueType &_value)
        {
            end = _end;
            value = _value;
        }
        int end;
        EdgeValueType value;
    };

    struct Node
    {
        int index;

        // these are only relevant when the node maps over an image domain
        int x, y;

        std::vector<Edge> edges;
    };

    OptGraph()
    {
        
    }

    OptGraph(int nodeCount)
    {
        allocate(nodeCount);
    }

    void allocate(int nodeCount)
    {
        nodes.resize(nodeCount);
        for (int n = 0; n < nodeCount; n++)
            nodes[n].index = n;
    }

    void finalize()
    {
        adjacencyOffsetsCPU.clear();
        adjacencyListsXCPU.clear();
        adjacencyListsYCPU.clear();
        edgeValuesCPU.clear();

        for (const Node &n : nodes)
        {
            adjacencyOffsetsCPU.push_back(adjacencyListsXCPU.size());
            for (const Edge &e : n.edges)
            {
                adjacencyListsXCPU.push_back(e.end);
                edgeValuesCPU.push_back(e.value);
            }
        }
        adjacencyOffsetsCPU.push_back(adjacencyListsXCPU.size());

        adjacencyListsYCPU = adjacencyListsXCPU;
        for (auto &x : adjacencyListsYCPU)
            x = 0;

        //cudaMalloc(&dataGPU, sizeof(float) * dimX * dimY);
    }

    void syncCPUToGPU() const
    {
        //cudaMemcpy(dataGPU, (void *)dataCPU.data(), sizeof(float) * dimX * dimY, cudaMemcpyHostToDevice);
    }

    void syncGPUToCPU() const
    {
        //cudaMemcpy((void *)dataCPU.data(), dataGPU, sizeof(float) * dimX * dimY, cudaMemcpyDeviceToHost);
    }

    std::vector<Node> nodes;

    std::vector<uint64_t> adjacencyOffsetsCPU;
    std::vector<uint64_t> adjacencyListsXCPU;
    std::vector<uint64_t> adjacencyListsYCPU;
    std::vector<EdgeValueType> edgeValuesCPU;
};

typedef OptGraph<float> OptGraphf;

//vector<uint64_t*> adjacencyOffsetsCPU;
//vector<uint64_t*> adjacencyListsCPU;
//vector<void*> edgeValuesCPU;

#endif
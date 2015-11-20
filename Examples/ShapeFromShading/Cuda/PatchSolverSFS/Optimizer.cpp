#include "stdafx.h"
#include "Optimizer.h"
#include <cstdio>
#include <stdint.h>
#include <vector>



Optimizer::Optimizer() : m_optimizerState(NULL), m_problem(NULL) {
    fprintf(stderr, "Opt_NewState()\n");
    m_optimizerState = Opt_NewState();
    if (m_optimizerState == nullptr)
    {
        fprintf(stderr, "Opt_NewState failed\n");
    }
}

Optimizer::~Optimizer() {
    if (m_problem != NULL) {
        Opt_ProblemDelete(m_optimizerState, m_problem);
    }
    // Free Opt State?
}

void Optimizer::defineProblem(std::string terraFile, std::string optimizerName) {
    fprintf(stderr, "Opt_ProblemDefine(%s, %s)\n", terraFile.c_str(), optimizerName.c_str());
    m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optimizerName.c_str(), NULL);
    if (m_problem == nullptr)
    {
        fprintf(stderr, "Opt_ProblemDefine failed\n");
    }
}

Plan* Optimizer::planProblem(int width, int height, const std::vector<uint32_t>& elemsize) {
    std::vector<uint32_t> stride;
    ;
    uint32_t dims[] = { width, height };
    for (int i = 0; i < elemsize.size(); ++i)
    {
        stride.push_back(width * elemsize[i]);
    }
    fprintf(stderr, "Opt_ProblemPlan(%ld, %ld)\n", dims[0], dims[1]);
    Plan * plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims, (uint32_t*)elemsize.data(), stride.data(), NULL, NULL, NULL);
    if (plan == nullptr)
    {
        fprintf(stderr, "Opt_ProblemPlan failed\n");
    }
    return plan;
}

struct IterStruct {
    unsigned int* nIter;
    unsigned int* lIter;
    unsigned int* pIter;
    IterStruct(unsigned int* n, unsigned int* l, unsigned int* p) : nIter(n), lIter(l), pIter(p) {}
};

void Optimizer::solve(Plan* plan, std::vector<void*> images, void* params) {
    unsigned int nIter[] = { 1, 10, 8 };
    IterStruct iterStruct(&nIter[0], &nIter[1], &nIter[2]);
    Opt_ProblemInit(m_optimizerState, plan, images.data(), NULL, (void**)params, (void**)&iterStruct);
    while (Opt_ProblemStep(m_optimizerState, plan, images.data(), NULL, (void**)params, (void**)&iterStruct));
}

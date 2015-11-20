extern "C" {
#include "../build/Opt.h"
}
#include <string>
class Optimizer {
    OptState *  m_optimizerState;
    Problem*    m_problem;
public:
    OptState* state() const {
        return m_optimizerState;
    }
    void defineProblem(std::string terraFile, std::string optimizerName);
    Plan* planProblem(int width, int height, const std::vector<uint32_t>& elemsize);
    void solve(Plan* plan, std::vector<void*> images, void* params);
    Optimizer();
    ~Optimizer();

};
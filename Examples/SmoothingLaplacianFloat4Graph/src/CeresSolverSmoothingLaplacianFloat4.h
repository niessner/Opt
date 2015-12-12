#pragma once

class CeresSolverSmoothingLaplacianFloat4 {
public:
    void solve(const ColorImageR32G32B32A32 &image, float weightFit, float weightReg, ColorImageR32G32B32A32 &result);

private:
    unsigned int _width, _height;

    vector<vec2i> makeImageEdges();
};

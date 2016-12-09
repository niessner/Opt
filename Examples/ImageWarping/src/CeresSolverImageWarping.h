#pragma once
#include "../../shared/Precision.h"
class CeresSolverWarping {

public:
    CeresSolverWarping(unsigned int width, unsigned int height)
	{
		m_width = width;
		m_height = height;

        h_x_double = new double2[width * height];
        h_a_double = new double[width * height];

        for (int i = 0; i < (int)width * (int)height; i++)
        {
            h_x_double[i].x = 0.0;
            h_x_double[i].y = 0.0;
            h_a_double[i] = 0.0;
        }
	}

    ~CeresSolverWarping()
	{
		
	}

    // !returns total time taken for solve, in milliseconds
    float solve(OPT_FLOAT2* h_x_float, OPT_FLOAT* h_a_float, OPT_FLOAT2* h_urshape, OPT_FLOAT2* h_constraints, OPT_FLOAT* h_mask, float weightFit, float weightReg, std::vector<SolverIteration>& results);

private:

	double2* h_x_double;
    double* h_a_double;
	int m_width, m_height;
};
#ifndef USE_CERES
inline float CeresSolverWarping::solve(OPT_FLOAT2* h_x_float, OPT_FLOAT* h_a_float, OPT_FLOAT2* h_urshape, OPT_FLOAT2* h_constraints, OPT_FLOAT* h_mask, float weightFit, float weightReg, std::vector<SolverIteration>& results)
{
    return 0.0f;
}
#endif
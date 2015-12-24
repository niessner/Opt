#pragma once

class CeresSolverWarping {

public:
    CeresSolverWarping(unsigned int width, unsigned int height)
	{
		m_width = width;
		m_height = height;

        h_x_double = new double2[width * height];
        h_a_double = new double[width * height];
	}

    ~CeresSolverWarping()
	{
		
	}

    void solve(float2* h_x_float, float* h_a_float, float2* h_urshape, float2* h_constraints, float* h_mask, float weightFit, float weightReg);

private:

	double2* h_x_double;
    double* h_a_double;
	unsigned int m_width, m_height;
};

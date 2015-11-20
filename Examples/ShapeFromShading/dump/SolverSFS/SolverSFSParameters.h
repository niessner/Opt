#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
	float weightFitting;					// Is initialized by the solver!
	float weightFittingStart;				// Starting value for incremental relaxation
	float weightFittingIncrement;			// Update factor

	float weightRegularizer;				// Regularization weight
	float weightPrior;						// Prior weight

	float weightShading;					// Shading weight
	float weightBoundary;					// Boundary weight

	unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
	unsigned int nLinIterations;			// Steps of the linear solver
};

#endif

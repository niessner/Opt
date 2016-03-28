typedef struct OptState OptState;
typedef struct Plan Plan;
typedef struct Problem Problem;

//allocate a new independant context for Opt
OptState * Opt_NewState();

// load the problem specification including the energy function from 'filename' and
// initializer a solver of type 'solverkind' (currently only one solver is supported: '')
Problem * Opt_ProblemDefine(OptState * state,const char * filename,const char * solverkind);
void Opt_ProblemDelete(OptState * state, Problem * problem);


// allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
// how the dimensions are used is based on the problem specification (see 'writing problem specifications')
Plan * Opt_ProblemPlan(OptState * state,Problem * problem, unsigned int * dimensions);
void Opt_PlanFree(OptState * state, Plan * plan);

// run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
// and outputs that define the problem, including images, graphs, and problem paramaters
// (see 'writing problem specifications'). 'solverparams' are the solver-specific parameter (e.g., 
//  number of iterations, see 'solver parameters')
void Opt_ProblemSolve(OptState * state,Plan * plan,void ** problemparams,void ** solverparams);

// use these two functions to control the outer solver loop on your own. In between iterations,
// problem parameters can be inspected and updated.

// run just the initialization for a problem, but do not do any outer steps.
void Opt_ProblemInit(OptState * state,Plan * plan,void ** problemparams,void ** solverparams);
// perform one outer iteration of the solver loop and return to the user.
// a zero return value indicates that the solver is finished according to its parameters
int Opt_ProblemStep(OptState * state,Plan * plan,void ** problemparams,void ** solverparams);
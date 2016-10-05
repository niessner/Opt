typedef struct Opt_State 	Opt_State;
typedef struct Opt_Plan 	Opt_Plan;
typedef struct Opt_Problem 	Opt_Problem;

//allocate a new independant context for Opt
Opt_State* Opt_NewState();

// load the problem specification including the energy function from 'filename' and
// initializer a solver of type 'solverkind' (currently only one solver is supported: '')
Opt_Problem* Opt_ProblemDefine(Opt_State* state, const char* filename, const char* solverkind);
void Opt_ProblemDelete(Opt_State* state, Opt_Problem* problem);


// allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
// how the dimensions are used is based on the problem specification (see 'writing problem specifications')
Opt_Plan* Opt_ProblemPlan(Opt_State* state, Opt_Problem* problem, unsigned int* dimensions);
void Opt_PlanFree(Opt_State * state, Opt_Plan* plan);

// run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
// and outputs that define the problem, including images, graphs, and problem paramaters
// (see 'writing problem specifications'). 'solverparams' are the solver-specific parameter (e.g., 
//  number of iterations, see 'solver parameters')
void Opt_ProblemSolve(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);

// use these two functions to control the outer solver loop on your own. In between iterations,
// problem parameters can be inspected and updated.

// run just the initialization for a problem, but do not do any outer steps.
void Opt_ProblemInit(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);
// perform one outer iteration of the solver loop and return to the user.
// a zero return value indicates that the solver is finished according to its parameters
int Opt_ProblemStep(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);
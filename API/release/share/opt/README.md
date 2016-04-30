==================
= Short Version ==
==================
Go to  \share\opt\examples\<folder> -> run the Makefile and/or VisualStudio solution -> run example
====================

=================
= Long Version ==
=================

This document describes how to use the Opt language (optlang.org). 

Opt is a new language in which a user simply writes energy functions over image- or graph-structured unknowns, and a compiler automatically generates state-of-the-art GPU optimization kernels. Real-world energy functions compile directly into highly optimized GPU solver implementations with performance competitive with the best published hand-tuned, application-specific GPU solvers.

This is an early release of the software to get feedback on the expressiveness of the language. We are interested in seeing what problems can be expressed and what features will be necessary to support more problems.

As an early release there are some things that are not complete that will be improved over time.

* Error reporting is limited and may be difficult to understand at times.
* Code can only run on NVIDIA GPUs with a relatively modern version of CUDA (6.5 or greater on OSX/Linux, and 7.0 on windows)
* The library of built-in math functions is somewhat limited. For instance, it include vectors but doesn't include small matrix operations.

These issues will improve over time, but if you run into issues, just send us an email:

* zdevito at cs dot stanford dot edu
* niessner at cs dot stanford dot edu

Overview
========

Opt is composed of a library `libOpt.a` (`Opt.lib` under windows) and a header file `Opt.h`. An application links Opt and uses its API to define and solve optimization problems. Opt's high-level energy functions behave like shaders in OpenGL. They are loaded as your application runs using the `Opt_ProblemDefine` API.

Our release includes:

* ./lib/libOpt.a (implementation of Opt)
* ./lib/libterra.a (a dependency of Opt)
* ./include/Opt.h (API that interfaces with your application)
* ./share/opt/examples/ImageWarping (a complete example application that uses Opt)
* ./share/opt/examples/MeshDeformation (a complete example application that uses graphs in Opt)

See the Makefiles in the examples for instructions on how to link Opt into your applications. In particular, on OSX, you will need to add the following linker flags:

    # osx only
    OSXFLAGS += -pagezero_size 10000 -image_base 100000000
    
    clang++ main.cpp -o main.cpp $(OSXFLAGS) -std=c++11 -L$(OPTHOME)/lib -L$(OPTHOME)/include -lOpt -lterra -ldl -pthread
    

Using the Opt C/C++ API
=======================

    OptState* Opt_NewState();
    
Allocate a new independant context for Opt
    
---
    
    Opt_Problem* Opt_ProblemDefine(Opt_State* state, const char* filename, const char* solverkind);

Load the energy specification from 'filename' and initialize a solver of type 'solverkind' (currently only one solver is supported: 'gaussNewtonGPU').
See writing energy specifications for how to describe energy functions.

---

    void Opt_ProblemDelete(Opt_State* state, Opt_Problem* problem);

Delete memory associated with the Problem object.

---

    Opt_Plan* Opt_ProblemPlan(Opt_State* state, Opt_Problem* problem, unsigned int* dimensions);

Allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
How the dimensions are used is based on the problem specification (see 'binding values' in 'writing energy specifications')

---

   void Opt_PlanFree(Opt_State * state, Opt_Plan* plan);

Delete the memory associated with the plan.

---

    Opt_ProblemSolve(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);

Run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
and outputs that define the problem, including arrays, graphs, and problem paramaters
(see 'writing problem specifications'). 'solverparams' are the solver-specific parameter (e.g., 
number of iterations, see 'solver parameters')

---

    void Opt_ProblemInit(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);
    int Opt_ProblemStep(Opt_State* state, Opt_Plan* plan, void** problemparams, void** solverparams);

Use these two functions to control the outer solver loop on your own. The arguments are the same as `Opt_ProblemSolve` but
the `Step` function returns between iterations of the solver. Problem parameters can be inspected and updated between calls to Step.
A zero return value indicates that the solver is finished according to its parameters.

    Opt_ProblemInit(...);
    while(Opt_ProblemStep(...) != 0) {
        // inspect and update problem state as desired.
    }
    // solver finished


Writing Energy Specifications
==============================

Specifications of the energy are written using an API embedded in Lua. 
Similar to SymPy or Mathematica, objects with overloaded operators in Lua are used to build up a symbolic expression of the energy. There are Lua functions to declare objects: dimensions, arrays (including unknown to be solved for), and graphs.

These objects can be used to create residuals functions defined per-pixel in an array or per-edge in graph. The mathematical expressions of energy are built using overloaded operators defined on these objects. The 'Energy' function adds an expression to the overall energy of the system. 

A simple laplacian smoothing energy in this system would have the form:


    W = Dim("W",0) 
    H = Dim("H",1)
    X = Array2D("X",float,{W,H},0) 
    A = Array2D("A",float,{W,H},1)
    
    w_fit,w_reg = .1,.9
    
    -- overloaded operators allow you to defined mathematical expressions as energies
    fit = w_fit*(X(0,0) - A(0,0))
    
    -- register fitting energy
    Energy(fit) --fitting
    
    -- Energy function can be called multiple times or with multiple arguments
    -- to add more residual terms to the energy
    Energy(w_reg*(X(0,0) - X(1,0)),
           w_reg*(X(0,0) - X(0,1)))


The functions are described in more details below.

## Declaring the inputs/outputs of an energies ##
 
    dimension = Dim(name,dimensions_position)
    
Create a new dimension used to describe the size of Arrays. `dimensions_position` is the 0-based offset into the `dimensions` argument to `Opt_ProblemPlan` that will be bound to this value. See 'Binding Values'.

    local W =
    H = Dim("W",0), Dim("H",1)
    
---

    array = Array(name,type,dimlist,problemparams_position)
    array = Unknown(name,type,dimlist,problemparams_position)
    
Declare a new input to the problem (`Array`), or an unknown value to be solved for `Unknown`. Both return an Array object that can be used to formulate energies.

`name` is the name of the object, used for debugging 
`type` can be float, float2, float3, ...
`dimlist` is a Lua array of dimensions (e.g., `{W,H}`). Arrays can be 1, 2, or 3 dimensional but 3 dims has not been tested heavily.
`problemparams_position` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that will be bound to this value. 

Examples:

    local Angle = Unknown("Angle",float, {W,H},1)
    local UrShape = Array("UrShape", float2,{W,H},2)	
    
---

    graph = Graph(name, problemparams_position_of_graph_size,
                 {vertexname, dimlist, problemparams_position_of_indices}*)

Declare a new graph that connects arrays together through hyper-edges.

`name` is a string for debugging.
`problemparams_position_of_graph_size` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that will determine the number of edges in the graph.

The remaining arguments are used to define vertices in the hyper-edge of the graph.
Each vertex requires the following arguments:

     vertexname, dimlist, problemparams_position_of_indices
     
`vertexname` is the name of the vertex used in the energy specification.
`dimlist` is a Lua array of dimensions (e.g., `{W,H}`). Arrays can be 1, 2, or 3 dimensional but 3 dims has not been tested heavily. This vertex will be a pointer into any array of this dimension.
`problemparams_position_of_indices` is the 0-based offset into the `problemparams` argument to `Opt_ProblemSolve` that is an array of indexes the size of the number of edges in the graph, where each entry is an index into the dimension specified in `dimlist`. For 2- or 3- dimensional arrays the indices for both dimensions are listed sequentially `(int,int)`.
    
Example:

    N = Dim("N",0)
    local Angle = Unknown("Angle", float3,{N},0)		
    local G =  Graph("G", 1, "head", {N}, 2,
                             "tail", {N}, 3)
                             
    Energy(Angle(G.v0) - Angle(G.v1))

---

## Writing Energies ##

Energies are described using a mathematical expressions constructed using Lua object overloaded.

Values can be read from the arrays created with the `Array` or `Unknown` constructors. 

### Accessing values with Stencils or Graphs ###

    value = Angle(0,0) -- value of the 'Angle' array at the centered pixel
    value = Angle(1,0) -- value of the 'Angle' array at the pixel to the right of the centered pixel
    value = Angle(0,2) -- value of the 'Angle' array at the pixel two pixels above the centered pixel
    ...

Each expression is implicitly defined over an entire array or entire set of edges. 
Expressions are implicitly squared and summed over all domains since our solver is for non-linear least squared problems. Energies are described per-pixel or per-edge with, e.g., `Angle(0,0)`, as the centered pixel. Other constant offsets can be given to select neighbors.

To access values at graph locations you use the name of the vertex as the index into the array:

    N = Dim("N",0)
    local Angle = Unknown("Angle", float3,{N},0)		
    local G =  Graph("G", 1, "head", {N}, 2,
                             "tail", {N}, 3)  

    value = Angle(G.head)
    value2 = Angle(G.tail)
    
### Math Operators ###

Generic math operators are usable on any value or vector:

    +
    -
    *
    /
    abs
    acos
    acosh
    and_
    asin
    asinh
    atan
    atan2
    classes
    cos
    cosh
    div
    eq
    exp
    greater
    greatereq
    less
    lesseq
    log
    log10
    mul
    not_
    or_
    pow
    prod
    sin
    sinh
    sqrt
    tan
    tanh
    Select(condition,truevalue,falsevalue) -- piecewise conditional operator, if condition ~= 0, it is truevalue, otherwise it is falsevalue
    scalar = All(vector) -- true if all values in the vector are true
    scalar = Any(vector) -- true of any value in the vector is true
	Rotate2D(angle, vector2)
	Rotate3D(angle3, vector3)
	
	

All operators apply elementwise to `Vector` objects.

Because Lua does not allow generic overloading of comparison ( `==` , '<=', ... ), you must use the functions we have provided instead for comparisions:
`eq(a,b)`, `lesseq(a,b)`, etc.


### Defining Energies ###

    `Energy(energy1,energy2,...)`
    
Add the terms `energy1`, ... to the energy of the whole problem. Energy terms are implicitly squared and summed over the entire domain (array or graph) on which they are defined.  Each channel of a `Vector` passed as an energy is treated as a separate energy term.


### Boundaries ###

For energies defined on arrays, it is possible to control how the energy behaves on the boundaries.  Any energy term has a particular pattern of data it reads from neighboring pixels in the arrays, which we call its `stencil`. By default, residual values are only defined for pixels in the array where the whole stencil has defined values. For a 3x3 stencil, for instance, this means that the 1-pixel border of an image will not evaluate this energy term (or equivalently, this term contributes 0 to the overall energy).

If you do not want the default behavior, you can use the `InBounds(x,y)` functions along with the `Select` function to describe custom behavior:

    customvalue = Select(InBounds(1,0),value_in_bounds,value_on_the_border) 

`InBounds` is true only when the relative offet `(1,0)` is in-bounds for the centered pixel. Any energy that uses `InBounds` will be evaluated at _every_ pixel including the border region, and it is up to the user to choose what to do about boundaries.

It is also possible to exclude arbitrary pixels from the solve using the `Exclude(exp)` method. When `exp` is true, unknowns defined at these pixels will not be updated and residuals at these pixels will not be evaluated.

### Vectors ###

    vector = Vector(a,b,c)
    vector2 = vector:dot(vector)
    scalar = vector:sum()
    numelements = vector:size() -- 3 for this vector
    vector3 = vector + vector -- elementwise addition

Objects of type `float2`, `float3` ... are vectors. The function `Vector` constructs them from individual elements. All math is done elementwise to vectors, including functions like `abs`.

### Binding Values for the C/C++ API ###

To connect values passed in from C/C++ API to values in the energy specification, the functions  `Array`, `Unknown`, `Dim`, and `Graph` have an argument (e.g., `problemparams_position`) that binds the object in the energy specification to the argument at that numeric offset in the parameters passed to Opt. 

API Example:

    uint32_t dims[] = { width, height };
	Plan * m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);
	
Energy Specification:

    local W,H = Dim("W",0), Dim("H",1)
    
API Example:

	void* solverParams[] = { &nNonLinearIterations, &nLinearIterations };
	float weightFitSqrt = sqrt(weightFit);
	float weightRegSqrt = sqrt(weightReg);
	
	float * d_x = ... //raw image data for x in (H,W,channel) order
	float * d_a = ...
	float * d_urshape = ...
	float * d_constraints = ...
	float * d_mask = ...
	
	void* problemParams[] = { d_x, d_a, d_urshape, d_constraints, d_mask, &weightFitSqrt, &weightRegSqrt };
		
	Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
    
Energy Specification:
    
    local Offset = Unknown("Offset",float2,{W,H},0)
    local Angle = Unknown("Angle",float,{W,H},1)
    local UrShape = 	Array("UrShape", float2,{W,H},2)		
    local Constraints = Array("Constraints", float2,{W,H},3)	
    local Mask = 		Array("Mask", float, {W,H},4)	
    local w_fitSqrt = Param("w_fitSqrt", float, 5)
    local w_regSqrt = Param("w_regSqrt", float, 6)


### Helpers ###

    for x,y in Stencil { {1,0},{-1,0},{0,1},{0,-1} } do
        Energy(X(0,0) - X(x,y)) -- laplacian regularization
    end
    
    -- equivalent to
    Energy(X(0,0) - X(1,0)) -- laplacian regularization
    Energy(X(0,0) - X(-1,0)) -- laplacian regularization
    Energy(X(0,0) - X(0,1)) -- laplacian regularization
    Energy(X(0,0) - X(0,-1)) -- laplacian regularization
    
The function `Stencil` is a Lua iterator that makes it easy to define similar energy functions for a set of neighboring offsets.


Solver Parameters
=================

The 'gaussNewtonGPU' solver takes two solver parameters passed in as an array of pointers:

parameter 0: the number of non-linear iterations
parameter 1: the number of linear iterations in each non-linear step

    int nonlinear = 2;
    int linear = 10;
    void * solverparams[] = { &nonlinear, linear};

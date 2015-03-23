extern "C" {
#include "Opt.h"
}
#include <stdio.h>

int main() {
    printf("this is main\n");
    OptState * S =  Opt_NewState();
    
    uint64_t dims[] = {1024,512};
    
    ImageBinding * A = Opt_ImageBind(S,NULL,1024,sizeof(double));
    ImageBinding * x = Opt_ImageBind(S,NULL,1,sizeof(double));
    ImageBinding * b = Opt_ImageBind(S,NULL,1,sizeof(double));
    
    Problem * prob = Opt_ProblemDefine(S,"src/example.t","gradientdescent",NULL);
    Plan * plan = Opt_ProblemPlan(S,prob,dims);
    
    ImageBinding * images[] = {x,A,b};
    
    Opt_ProblemSolve(S,plan,images,NULL);
}
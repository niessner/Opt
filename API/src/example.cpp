extern "C" {
#include "Opt.h"
}
#include <stdio.h>

int main() {
    printf("this is main\n");
    OptState * S =  Opt_NewState();
    ImageBinding * binding = Opt_ImageBind(S,NULL,1024,sizeof(float));
    Problem * prob = Opt_ProblemDefine(S,"src/example.t");
    Plan * plan = Opt_ProblemPlan(S,prob,"gradientdescent",NULL);
    Opt_ProblemSolve(S,plan,binding);
}
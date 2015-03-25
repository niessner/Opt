
#include "main.h"

int main()
{
  //printf("Hello\n");
    TestFramework test;
    test.runAllTests();
    return 0;
    /*OptState * S =  Opt_NewState();
    uint64_t dims[] = {1024,512};
    
    ImageBinding * A = Opt_ImageBind(S,NULL,1024,sizeof(double));
    ImageBinding * x = Opt_ImageBind(S,NULL,1,sizeof(double));
    ImageBinding * b = Opt_ImageBind(S,NULL,1,sizeof(double));
    
    cout << "start Opt_ProblemDefine" << endl;
    Problem * prob = Opt_ProblemDefine(S,"example.t","gradientdescent",NULL);

    if (!prob)
    {
        cout << "Opt_ProblemDefine failed" << endl;
        cin.get();
        return 1;
    }
    
    Plan * plan = Opt_ProblemPlan(S,prob,dims);
    
    ImageBinding * images[] = {x,A,b};
    
    Opt_ProblemSolve(S,plan,images,NULL);*/

}

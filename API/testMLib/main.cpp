
#include "main.h"

int main(int argc, char ** argv)
{
  //printf("Hello\n");
    TestFramework test;
    test.runAllTests(argc, argv);
    cout << "done" << endl;
    #ifdef _WIN32
    cin.get();
    #endif
    return 0;
    /*OptState * S =  Opt_NewState();
    uint64_t dims[] = {1024,512};
    
    ImageBinding * A = Opt_ImageBind(S,NULL,1024,sizeof(float));
    ImageBinding * x = Opt_ImageBind(S,NULL,1,sizeof(float));
    ImageBinding * b = Opt_ImageBind(S,NULL,1,sizeof(float));
    
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

#include <stdint.h>
typedef struct OptState OptState;
typedef struct Plan Plan;
typedef struct Problem Problem;
OptState * Opt_NewState();
void Opt_PlanFree(OptState *,Plan *);
Problem * Opt_ProblemDefine(OptState *,const char *,const char *,void *);
void Opt_ProblemDelete(OptState *,Problem *);
void Opt_ProblemInit(OptState *,Plan *,void * *,void * *,void * *,void * *);
Plan * Opt_ProblemPlan(OptState *,Problem *,uint32_t *,uint32_t *,uint32_t *,int32_t * *,int32_t * *,int32_t * *);
void Opt_ProblemSolve(OptState *,Plan *,void * *,void * *,void * *,void * *);
int32_t Opt_ProblemStep(OptState *,Plan *,void * *,void * *,void * *,void * *);

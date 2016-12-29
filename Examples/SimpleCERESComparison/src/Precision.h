#pragma once

// To change Opt's precision:
// set OPT_DOUBLE_PRECISION to 0 or 1
// opt_precision.t true/false
// precision.t double/float

#define OPT_DOUBLE_PRECISION 1


struct double9 {
    double vals[9];
};

struct float9 {
    float vals[9];
};

#if OPT_DOUBLE_PRECISION
#   define OPT_FLOAT double
#   define OPT_FLOAT2 double2
#   define OPT_FLOAT3 double3
#   define OPT_FLOAT4 double4
#   define OPT_FLOAT9 double9
#else
#   define OPT_FLOAT float
#   define OPT_FLOAT2 float2
#   define OPT_FLOAT3 float3
#   define OPT_FLOAT4 float4
#   define OPT_FLOAT9 float9
#endif
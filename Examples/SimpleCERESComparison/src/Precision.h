#pragma once

// To change Opt's precision:
// set OPT_DOUBLE_PRECISION to 0 or 1
// opt_precision.t true/false
// precision.t double/float

#define OPT_DOUBLE_PRECISION 1

#if OPT_DOUBLE_PRECISION
#   define OPT_FLOAT double
#   define OPT_FLOAT2 double2
#   define OPT_FLOAT3 double3
#   define OPT_FLOAT4 double4
#else
#   define OPT_FLOAT float
#   define OPT_FLOAT2 float2
#   define OPT_FLOAT3 float3
#   define OPT_FLOAT4 float4
#endif
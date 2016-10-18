#pragma once

#define OPT_DOUBLE_PRECISION 1

#if OPT_DOUBLE_PRECISION
#   define OPT_FLOAT double
#   define OPT_FLOAT2 double2
#else
#   define OPT_FLOAT float
#   define OPT_FLOAT2 float2
#endif
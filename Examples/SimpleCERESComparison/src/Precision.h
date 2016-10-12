#pragma once

// NYI, do not change from 0 without implementing!
#define OPT_DOUBLE_PRECISION 0

#if OPT_DOUBLE_PRECISION
#   define OPT_FLOAT double
#else
#   define OPT_FLOAT float
#endif
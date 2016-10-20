
#pragma once

// exactly one of these should be set
const bool useProblemDefault = false; //2 unknowns -- Michael's original curveFitting.t code
const bool useProblemMisra = false;   //2 unknowns
const bool useProblemBennet5 = true;  //3 unknowns

const double3 misraSolution = { 2.3894212918E+02, 5.5015643181E-04, 0.0 };
const double3 bennet5Solution = { -2.5235e3, 4.6736e1, 9.32184e-1 };

const int unknownCount = 3;

//#define OPT_UNKNOWNS OPT_FLOAT2
#define OPT_UNKNOWNS OPT_FLOAT3

//#define UNKNOWNS double2
#define UNKNOWNS double3
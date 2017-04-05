import 'ebb'
local L = require 'ebblib'

local GridLib   = require 'ebb.domains.grid'

local vdb       = require 'ebb.lib.vdb'
local CSV       = require 'ebb.io.csv'
local W = 512
local H = 512

local grid = GridLib.NewGrid2d {
  size   = {W, H},
  origin = {0, 0},
  width  = {W, H},
  periodic_boundary = {false,false},
}

local FLOAT_EPSILON     = L.Constant(L.float, 0.0001)

local weightFitting    = L.Constant(L.float, 100.0)
local weightRegularizer  = L.Constant(L.float, 0.01)
local scanAlphaNumerator = L.Global(L.float, 0.0)
local scanAlphaDenominator = L.Global(L.float, 0.0)
local scanBetaNumerator = L.Global(L.float, 0.0)
local cost = L.Global(L.float, 0.0)


grid.cells:NewField('X', L.vec2f):Load(CSV.Load, "urshape.csv")
grid.cells:NewField('A', L.float):Load(0)
grid.cells:NewField('mask', L.float):Load(CSV.Load, "mask.csv")
grid.cells:NewField('constraint', L.vec2f):Load(CSV.Load, "constraints.csv")
grid.cells:NewField('urshape', L.vec2f):Load(CSV.Load, "urshape.csv")

local ebb mul(M : L.mat2f,v : L.vec2f)
  return {M[0,0]*v[0] + M[1,0]*v[1], M[0,1]*v[0] + M[1,1]*v[1]}
end
local ebb dot(v0 : L.vec2f,v1 : L.vec2f)
  return {v0[0]*v1[0], v0[1]*v1[1]}
end
local ebb evalR(angle : L.float)
    var cosA : L.float = L.float(L.cos(angle))
    var sinA : L.float = L.float(L.sin(angle))
    var R = {{ cosA, -sinA },
    { sinA, cosA }}
    return R
end
--[[ Rotation Matrix dAlpha
inline __device__ float2x2 eval_dR(float CosAlpha, float SinAlpha)
{
  float2x2 R;
  R.m11 = -SinAlpha; R.m12 = -CosAlpha;
  R.m21 = CosAlpha;  R.m22 = -SinAlpha;
  return R;
}
--]]
local ebb evalR_dR(angle : L.float)
    var cosA : L.float = L.float(L.cos(angle))
    var sinA : L.float = L.float(L.sin(angle))
    var R = {{ -sinA, -cosA },
    { cosA, sinA }}
    return R
end

--[[
__inline__ __device__ float2 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float& bA)
{
  state.d_delta[variableIdx] = make_float2(0.0f, 0.0f);
  state.d_deltaA[variableIdx] = 0.0f;

  float2 b = make_float2(0.0f, 0.0f);
  bA = 0.0f;

  float2 pre = make_float2(0.0f, 0.0f);
  float preA = 0.0f;

  int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
  const int n0_i = i;   const int n0_j = j - 1; 
  const int n1_i = i;   const int n1_j = j + 1; 
  const int n2_i = i - 1; const int n2_j = j;   
  const int n3_i = i + 1; const int n3_j = j;   


  const bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height) && state.d_mask[get1DIdx(n0_i, n0_j, input.width, input.height)] == 0;
  const bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height) && state.d_mask[get1DIdx(n1_i, n1_j, input.width, input.height)] == 0;
  const bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height) && state.d_mask[get1DIdx(n2_i, n2_j, input.width, input.height)] == 0;
  const bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height) && state.d_mask[get1DIdx(n3_i, n3_j, input.width, input.height)] == 0;

  const bool b_ = isInsideImage(i   , j   , input.width, input.height);
  const bool b0 = isInsideImage(n0_i, n0_j, input.width, input.height) && b_;
  const bool b1 = isInsideImage(n1_i, n1_j, input.width, input.height) && b_;
  const bool b2 = isInsideImage(n2_i, n2_j, input.width, input.height) && b_;
  const bool b3 = isInsideImage(n3_i, n3_j, input.width, input.height) && b_;

  const bool m  = state.d_mask[get1DIdx(i   , j   , input.width, input.height)] == 0;
  const bool m0 = validN0;
  const bool m1 = validN1;
  const bool m2 = validN2;
  const bool m3 = validN3;


  // fit/pos
  float2 constraintUV = input.d_constraints[variableIdx]; bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && state.d_mask[get1DIdx(i, j, input.width, input.height)] == 0;
  if (validConstraint) { b += -2.0f*parameters.weightFitting*(state.d_x[variableIdx] - constraintUV); pre += 2.0f*parameters.weightFitting*make_float2(1.0f, 1.0f); }

  // reg/pos
  float2   p = state.d_x[get1DIdx(i, j, input.width, input.height)];
  float2   pHat = state.d_urshape[get1DIdx(i, j, input.width, input.height)];
  float2x2 R_i = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
  float2 e_reg = make_float2(0.0f, 0.0f);

  if (b0) { 
    float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; 
    float2x2 R_j = evalR(state.d_A[get1DIdx(n0_i, n0_j, input.width, input.height)]); 
    if (m0) {
      e_reg += (p - q) - float2(mat2x2(R_i)*mat2x1(pHat - qHat)); 
      pre += 2.0f*parameters.weightRegularizer; 
    }
    if (m) {
      e_reg += (p - q) - float2(mat2x2(R_j)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
  }
  if (b1) { 
    float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; 
    float2x2 R_j = evalR(state.d_A[get1DIdx(n1_i, n1_j, input.width, input.height)]); 
    if (m1) {
      e_reg += (p - q) - float2(mat2x2(R_i)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
    if (m) {
      e_reg += (p - q) - float2(mat2x2(R_j)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
  }
  if (b2) { 
    float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; 
    float2x2 R_j = evalR(state.d_A[get1DIdx(n2_i, n2_j, input.width, input.height)]); 
    if (m2) {
      e_reg += (p - q) - float2(mat2x2(R_i)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
    if (m) {
      e_reg += (p - q) - float2(mat2x2(R_j)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
  }
  if (b3) { 
    float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; 
    float2x2 R_j = evalR(state.d_A[get1DIdx(n3_i, n3_j, input.width, input.height)]); 
    if (m3) {
      e_reg += (p - q) - float2(mat2x2(R_i)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
    if (m) {
      e_reg += (p - q) - float2(mat2x2(R_j)*mat2x1(pHat - qHat));
      pre += 2.0f*parameters.weightRegularizer;
    }
  }
  b += -2.0f*parameters.weightRegularizer*e_reg;

  // reg/angle
  float2x2 R = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
  float2x2 dR = evalR_dR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
  float e_reg_angle = 0.0f;

  if (validN0) { 
    float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)];
    mat2x1 D = -mat2x1(dR*(pHat - qHat)); 
    e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); 
    preA += D.getTranspose()*D*parameters.weightRegularizer; 
  }

  if (validN1) { 
    float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; 
    mat2x1 D = -mat2x1(dR*(pHat - qHat)); 
    e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); 
    preA += D.getTranspose()*D*parameters.weightRegularizer; 
  }
  
  if (validN2) { 
    float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; 
    mat2x1 D = -mat2x1(dR*(pHat - qHat)); 
    e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); 
    preA += D.getTranspose()*D*parameters.weightRegularizer; 
  }
  
  if (validN3) { 
    float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; 
    float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; 
    mat2x1 D = -mat2x1(dR*(pHat - qHat)); 
    e_reg_angle += D.getTranspose()*mat2x1((p - q) - R*(pHat - qHat)); 
    preA += D.getTranspose()*D*parameters.weightRegularizer; 
  }

  preA = 2.0f*preA;
  bA += -2.0f*parameters.weightRegularizer*e_reg_angle;


  //pre = make_float2(1.0f, 1.0f);
  //preA = 1.0f;

  // Preconditioner
  if (pre.x > FLOAT_EPSILON) pre = 1.0f / pre;
  else               pre = make_float2(1.0f, 1.0f);
  state.d_precondioner[variableIdx] = pre;

  // Preconditioner
  if (preA > FLOAT_EPSILON) preA = 1.0f / preA;
  else            preA = 1.0f;
  state.d_precondionerA[variableIdx] = preA;
  

  return b;
}
--]]
local ebb applyJTF( c : grid.cells )

  c.delta = {0.0, 0.0}
  c.deltaA = 0.0

  var b  : L.vec2f = {0.0, 0.0}
  var bA : L.float = 0.0

  var pre : L.vec2f = {0.0, 0.0}
  var preA : L.float = 0.0

  var b0 = (c.yneg_depth == 0)
  var b1 = (c.ypos_depth == 0)
  var b2 = (c.xneg_depth == 0)
  var b3 = (c.xpos_depth == 0)


  -- fit/pos
  var validConstraint = (c.constraint[0] >= 0 and c.constraint[1] >= 0) and c.mask == 0
  if (validConstraint) then
    b += -2.0f*weightFitting*(c.X - c.constraint)
    pre += 2.0f*weightFitting*{1.0f, 1.0f}
  end
  -- reg/pos
  var R_i = evalR(c.A)
  var e_reg : L.vec2f = {0.0, 0.0}

  if b0 then 
    var c_n = c(0,-1)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
  end
  if b1 then 
    var c_n = c(0,1)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
  end
  if b2 then 
    var c_n = c(-1,0)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
  end
  if b3 then 
    var c_n = c(1,0)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*weightRegularizer
    end
  end
  b += -2.0f*weightRegularizer*e_reg

  -- reg/angle
  var R : L.mat2f = evalR(c.A)
  var dR : L.mat2f  = evalR_dR(c.A)
  var e_reg_angle : L.float = 0.0f
  var preA : L.float = 0.0f
  if b0 and (c(0,-1).mask == 0) then 
    var c_n = c(0,-1)
    var D = -mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,(c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape)))
    preA += weightRegularizer*dot(D,D)
  end
  if b1 and (c(0,1).mask == 0) then 
    var c_n = c(0,1)
    var D = -mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,(c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape)))
    preA += weightRegularizer*dot(D,D)
  end
  if b2 and (c(-1,0).mask == 0) then 
    var c_n = c(-1,0)
    var D = -mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,(c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape)))
    preA += weightRegularizer*dot(D,D)
  end
  if b3 and (c(1,0).mask == 0) then 
    var c_n = c(1,0)
    var D = -mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,(c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape)))
    preA += weightRegularizer*dot(D,D)
  end

  preA = 2.0f*preA
  bA += -2.0f*weightRegularizer*e_reg_angle

  -- Preconditioner
  if (pre.x > FLOAT_EPSILON) then 
    pre[0] = 1.0f / pre[0]
    pre[1] = 1.0f / pre[1]
  else 
    pre[0] = 1.0f
    pre[1] = 1.0f
  end
  c.pre = pre

  -- Preconditioner
  if (preA > FLOAT_EPSILON) then 
    preA = 1.0f / preA
  else 
    preA = 1.0f
  end
  c.preA = preA
  
  var result : L.vec3f = {b[0],b[1],bA}
  return result
end



--[[
__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
  const unsigned int N = input.N;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  float d = 0.0f;
  if (x < N)
  {
    float residuumA;
    const float2 residuum = evalMinusJTFDevice(x, input, state, parameters, residuumA); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
    state.d_r[x]  = residuum;                        // store for next iteration
    state.d_rA[x] = residuumA;                         // store for next iteration

    const float2 p  = state.d_precondioner[x]  * residuum;           // apply preconditioner M^-1
    state.d_p[x] = p;

    const float pA = state.d_precondionerA[x] * residuumA;           // apply preconditioner M^-1
    state.d_pA[x] = pA;

    d = dot(residuum, p) + residuumA * pA;                 // x-th term of nomimator for computing alpha and denominator for computing beta
  }

    d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}
--]]
local ebb PCGInit1(c : grid.cells)
  var d : L.float = 0.0 -- init for out of bounds lanes
  var residuum : L.vec3f = 0.0f

  var residuumFull = applyJTF(c)-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
  c.r  = {residuumFull[0],residuumFull[1]}  -- store for next iteration
  c.rA = residuumFull[2] -- store for next iteration

  var p  = {c.pre[0] * c.r[0], c.pre[1] * c.r[1]} -- apply preconditioner M^-1
  c.p = p

  var pA = c.preA * c.rA -- apply preconditioner M^-1
  c.d_pA = pA

  scanAlphaNumerator += dot(residuum, p) + residuumA * pA

end

local function PCGStep1()

end

local function PCGStep2()

end

local function PCGStep3()

end

local function PCGLinearUpdate()

end

--[[
// Rotation Matrix
inline __device__ float2x2 evalR(float CosAlpha, float SinAlpha)
{
  float2x2 R;
  R.m11 = CosAlpha; R.m12 = -SinAlpha;
  R.m21 = SinAlpha; R.m22 =  CosAlpha;
  return R;
}

inline __device__ float2x2 evalR(float& angle)
{
  return evalR(cos(angle), sin(angle));
}
--]]

local ebb measure_cost( c : grid.cells )
--[[
  float2 e = make_float2(0.0f, 0.0f);

  int i; int j; get2DIdx(variableIdx, input.width, input.height, i, j);
  const int n0_i = i;   const int n0_j = j - 1; bool validN0 = isInsideImage(n0_i, n0_j, input.width, input.height); if(validN0) { validN0 = (state.d_mask[get1DIdx(n0_i, n0_j, input.width, input.height)] == 0); };
  const int n1_i = i;   const int n1_j = j + 1; bool validN1 = isInsideImage(n1_i, n1_j, input.width, input.height); if(validN1) { validN1 = (state.d_mask[get1DIdx(n1_i, n1_j, input.width, input.height)] == 0); };
  const int n2_i = i - 1; const int n2_j = j;   bool validN2 = isInsideImage(n2_i, n2_j, input.width, input.height); if(validN2) { validN2 = (state.d_mask[get1DIdx(n2_i, n2_j, input.width, input.height)] == 0); };
  const int n3_i = i + 1; const int n3_j = j;   bool validN3 = isInsideImage(n3_i, n3_j, input.width, input.height); if(validN3) { validN3 = (state.d_mask[get1DIdx(n3_i, n3_j, input.width, input.height)] == 0); };

  // E_fit
  float2 constraintUV = input.d_constraints[variableIdx]; bool validConstraint = (constraintUV.x >= 0 && constraintUV.y >= 0) && state.d_mask[get1DIdx(i, j, input.width, input.height)] == 0;
  if (validConstraint) { 
    float2 e_fit = (state.d_x[variableIdx] - constraintUV); 
    e += parameters.weightFitting*e_fit*e_fit; 
  }

  // E_reg
  float2x2 R = evalR(state.d_A[get1DIdx(i, j, input.width, input.height)]);
  float2   p = state.d_x[get1DIdx(i, j, input.width, input.height)];
  float2   pHat = state.d_urshape[get1DIdx(i, j, input.width, input.height)];
  float2 e_reg = make_float2(0.0f, 0.0f);
  if (validN0) { float2 q = state.d_x[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n0_i, n0_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
  if (validN1) { float2 q = state.d_x[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n1_i, n1_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
  if (validN2) { float2 q = state.d_x[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n2_i, n2_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
  if (validN3) { float2 q = state.d_x[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 qHat = state.d_urshape[get1DIdx(n3_i, n3_j, input.width, input.height)]; float2 d = (p - q) - R*(pHat - qHat); e_reg += d*d; }
  e += parameters.weightRegularizer*e_reg;

  float res = e.x + e.y;
  return res;
--]]
  var e : L.vec2f = {0,0}
  -- E_fit
  var validConstraint = (c.constraint[0] >= 0 and c.constraint[1] >= 0) and c.mask == 0
  if (validConstraint) then
    var e_fit = (c.X - c.constraint)
    e[0] += weightFitting*e_fit[0]*e_fit[0]
    e[1] += weightFitting*e_fit[1]*e_fit[1]
  end

  var e_reg : L.vec2f = {0,0}
  var R : L.mat2f = evalR(c.A)
  if c.yneg_depth == 0 then 
    var c_n = c(0,-1)
    if c_n.mask == 0 then
      var d = (c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape))
      e_reg[0] += d[0]*d[0]
      e_reg[1] += d[1]*d[1]
    end
  end
  if c.ypos_depth == 0 then 
    var c_n = c(0,1)
    if c_n.mask == 0 then
      var d = (c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape))
      e_reg[0] += d[0]*d[0]
      e_reg[1] += d[1]*d[1]
    end
  end
  if c.xneg_depth == 0 then 
    var c_n = c(-1,0)
    if c_n.mask == 0 then
      var d = (c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape))
      e_reg[0] += d[0]*d[0]
      e_reg[1] += d[1]*d[1]
    end
  end
  if c.xpos_depth == 0 then 
    var c_n = c(1,0)
    if c_n.mask == 0 then
      var d = (c.X - c_n.X) - mul(R,(c.urshape - c_n.urshape))
      e_reg[0] += d[0]*d[0]
      e_reg[1] += d[1]*d[1]
    end
  end
  e += weightRegularizer*e_reg
  cost += e[0] + e[1]
end

local function computeCost()
  cost:set(0)
  grid.cells:foreach(measure_cost)
end


--grid.cells.X:Dump(CSV.Dump, "dump.csv")
local nIterations = 500
local lIterations = 500
for nIter=0, nIterations do
  scanAlphaNumerator:set(0)
  scanAlphaDenominator:set(0)
  scanBetaNumerator:set(0)
  grid.cells:foreach(PCGInit1)
  for lIter = 0, lIterations do       
    scanAlphaDenominator:set(0)           
    PCGStep1()
    scanBetaNumerator:set(0)
    PCGStep2()
    PCGStep3()
    -- save new rDotz for next iteration
    scanAlphaNumerator:set(scanBetaNumerator:get())
  end
  PCGLinearUpdate()
  computeCost()
  print( 'iteration #'..tostring(nIter), 'cost: ', tostring(cost:get()*0.5) )
end


--[[ GN-solver
  for  nIter=0, pd.solverparameters.nIterations do
    C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float)) --scan in PCGInit1 requires reset
    C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float)) --scan in PCGInit1 requires reset
    C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))  --scan in PCGInit1 requires reset
    gpu.PCGInit1(pd)
    if isGraph then
      gpu.PCGInit1_Graph(pd)  
      gpu.PCGInit1_Finish(pd) 
    end
    for lIter = 0, pd.solverparameters.lIterations do       
      C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))              
      gpu.PCGStep1(pd)
      if isGraph then
          gpu.PCGStep1_Graph(pd)
      end
      C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
      gpu.PCGStep2(pd)
      gpu.PCGStep3(pd)

      -- save new rDotz for next iteration
      C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)  
    end
  

    gpu.PCGLinearUpdate(pd)    
    var newCost = computeCost(pd)

    pd.prevCost = newCost 
    pd.solverparameters.nIter = pd.solverparameters.nIter + 1
  end
end
--]]


--[[
for i=1,360 do
  grid.cells.interior:foreach(update_temperature)
  grid.cells.boundary:foreach(update_temp_boundaries)
  grid.cells:Swap('t', 'new_t')

  vdb.vbegin()
  vdb.frame()
    grid.cells:foreach(visualize)
  vdb.vend()

  if i % 10 == 0 then -- measure statistics every 10 steps
    max_diff:set(0)
    grid.cells.interior:foreach(measure_max_diff)
    print( 'iteration #'..tostring(i), 'max gradient: ', max_diff:get() )
  end
end
-]]
-- Our simulation loop is mostly the same, but with one major difference.
-- Rather than run `update_temperature` for each cell, we only run it for
-- each `interior` cell.  Likewise, we then execute the boundary computation
-- only for each boundary cell.  Though we still visualize all the cells
-- with a single call.  (Note that if we ran `update_temperature` on all of
-- the cells, then we would produce array out of bound errors.)



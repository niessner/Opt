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


grid.cells:NewField('delta', L.vec2f):Load({0,0})
grid.cells:NewField('pre', L.vec2f):Load({0,0})
grid.cells:NewField('r', L.vec2f):Load({0,0})
grid.cells:NewField('p', L.vec2f):Load({0,0})
grid.cells:NewField('z', L.vec2f):Load({0,0})
grid.cells:NewField('Ap_X', L.vec2f):Load({0,0})

grid.cells:NewField('deltaA', L.float):Load(0.0)
grid.cells:NewField('preA', L.float):Load(0.0)
grid.cells:NewField('rA', L.float):Load(0.0)
grid.cells:NewField('pA', L.float):Load(0.0)
grid.cells:NewField('zA', L.float):Load(0.0)
grid.cells:NewField('Ap_XA', L.float):Load(0.0)

local ebb mul(M : L.mat2f,v : L.vec2f)
  return {M[0,0]*v[0] + M[1,0]*v[1], M[0,1]*v[0] + M[1,1]*v[1]}
end
local ebb mulV2(v0 : L.vec2f, v1 : L.vec2f)
  return {v0[0]*v1[0] , v0[1]*v1[1]}
end
local ebb dot(v0 : L.vec2f, v1 : L.vec2f)
  return v0[0]*v1[0] + v0[1]*v1[1]
end

local ebb evalR(angle : L.float)
    var cosA : L.float = L.float(L.cos(angle))
    var sinA : L.float = L.float(L.sin(angle))
    var R = {{ cosA, -sinA },
    { sinA, cosA }}
    return R
end

local ebb evalR_dR(angle : L.float)
    var cosA : L.float = L.float(L.cos(angle))
    var sinA : L.float = L.float(L.sin(angle))
    var R = {{ -sinA, -cosA },
    { cosA, sinA }}
    return R
end

local ebb applyJTF( c : grid.cells )

  c.delta = {0.0f, 0.0f}
  c.deltaA = 0.0f

  var b  : L.vec2f = {0.0f, 0.0f}
  var bA : L.float = 0.0f

  var pre : L.vec2f = {0.0f, 0.0f}
  var preA : L.float = 0.0f

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
  var e_reg : L.vec2f = {0.0f, 0.0f}
  var wReg : L.vec2f = {weightRegularizer, weightRegularizer}
  if b0 then 
    var c_n = c(0,-1)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
  end
  if b1 then 
    var c_n = c(0,1)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
  end
  if b2 then 
    var c_n = c(-1,0)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
  end
  if b3 then 
    var c_n = c(1,0)
    if c_n.mask == 0 then
      e_reg += (c.X - c_n.X) - mul(R_i,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
    end
    if c.mask == 0 then
      var R_j = evalR(c_n.A)
      e_reg += (c.X - c_n.X) - mul(R_j,(c.urshape - c_n.urshape))
      pre += 2.0f*wReg
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
  if (pre[0] > FLOAT_EPSILON) then 
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

local ebb applyJTJ( c : grid.cells )
  var b  : L.vec2f = {0.0f, 0.0f}
  var bA : L.float = 0.0f

  var b0 = (c.yneg_depth == 0)
  var b1 = (c.ypos_depth == 0)
  var b2 = (c.xneg_depth == 0)
  var b3 = (c.xpos_depth == 0)


  -- pos/constraint
  var validConstraint = (c.constraint[0] >= 0 and c.constraint[1] >= 0) and c.mask == 0
  if validConstraint then
    b += -2.0f*weightFitting*c.p
  end
  -- pos/reg
  var e_reg : L.vec2f = {0.0f, 0.0f}
  if b0 then
    var c_n = c(0,-1)
    if c.mask == 0    then e_reg += (c.p - c_n.p) end
    if c_n.mask == 0  then e_reg += (c.p - c_n.p) end
  end
  if b1 then
    var c_n = c(0,1)
    if c.mask == 0    then e_reg += (c.p - c_n.p) end
    if c_n.mask == 0  then e_reg += (c.p - c_n.p) end
  end
  if b2 then
    var c_n = c(-1,0)
    if c.mask == 0    then e_reg += (c.p - c_n.p) end
    if c_n.mask == 0  then e_reg += (c.p - c_n.p) end
  end
  if b3 then
    var c_n = c(1,0)
    if c.mask == 0    then e_reg += (c.p - c_n.p) end
    if c_n.mask == 0  then e_reg += (c.p - c_n.p) end
  end
  
  b += 2.0f*weightRegularizer*e_reg

  -- angle/reg
  var e_reg_angle : L.float = 0.0f
  var dR : L.mat2f = evalR_dR(c.A)
  if b0 and (c(0,-1).mask == 0) then 
    var D = mul(dR,(c.urshape - c(0,-1).urshape))
    e_reg_angle += dot(D,D)*c.pA
  end
  if b1 and (c(0,1).mask == 0) then 
    var D = mul(dR,(c.urshape - c(0,1).urshape))
    e_reg_angle += dot(D,D)*c.pA
  end
  if b2 and (c(-1,0).mask == 0) then 
    var D = mul(dR,(c.urshape - c(-1,0).urshape))
    e_reg_angle += dot(D,D)*c.pA
  end
  if b3 and (c(1,0).mask == 0) then 
    var D = mul(dR,(c.urshape - c(1,0).urshape))
    e_reg_angle += dot(D,D)*c.pA
  end
  bA += 2.0f*weightRegularizer*e_reg_angle


  -- upper right block
  e_reg = {0.0f, 0.0f}
  if b0 then
    var c_n = c(0,-1) 
    var dR_j = evalR_dR(c_n.A)
    var D    = -mul(dR,(c.urshape - c_n.urshape))
    var  D_j = mul(dR_j,(c.urshape - c_n.urshape))
    if c_n.mask == 0 then e_reg += D*c.pA end
    if c.mask == 0 then e_reg -= D_j*c_n.pA end
  end
  if b1 then
    var c_n = c(0,1) 
    var dR_j = evalR_dR(c_n.A)
    var D    = -mul(dR,(c.urshape - c_n.urshape))
    var  D_j = mul(dR_j,(c.urshape - c_n.urshape))
    if c_n.mask == 0 then e_reg += D*c.pA end
    if c.mask == 0 then e_reg -= D_j*c_n.pA end
  end
  if b2 then
    var c_n = c(-1,0) 
    var dR_j = evalR_dR(c_n.A)
    var D    = -mul(dR,(c.urshape - c_n.urshape))
    var  D_j = mul(dR_j,(c.urshape - c_n.urshape))
    if c_n.mask == 0 then e_reg += D*c.pA end
    if c.mask == 0 then e_reg -= D_j*c_n.pA end
  end
  if b3 then
    var c_n = c(1,0) 
    var dR_j = evalR_dR(c_n.A)
    var D    = -mul(dR,(c.urshape - c_n.urshape))
    var  D_j = mul(dR_j,(c.urshape - c_n.urshape))
    if c_n.mask == 0 then e_reg += D*c.pA end
    if c.mask == 0 then e_reg -= D_j*c_n.pA end
  end

  b += 2.0f*weightRegularizer*e_reg

  -- lower left block
  e_reg_angle = 0.0f
  if b0 and (c(0,-1).mask == 0) then
    var c_n = c(0,-1)
    var D = mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,c.p-c_n.p)
  end
  if b1 and (c(0,1).mask == 0) then
    var c_n = c(0,1)
    var D = mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,c.p-c_n.p)
  end
  if b2 and (c(-1,0).mask == 0) then
    var c_n = c(-1,0)
    var D = mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,c.p-c_n.p)
  end
  if b3 and (c(1,0).mask == 0) then
    var c_n = c(1,0)
    var D = mul(dR,(c.urshape - c_n.urshape))
    e_reg_angle += dot(D,c.p-c_n.p)
  end
  bA += 2.0f*weightRegularizer*e_reg_angle

  var result : L.vec3f = {b[0],b[1],bA}
  return result
end  

local ebb PCGInit1(c : grid.cells)
  var residuumFull = applyJTF(c)-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
  c.r  = {residuumFull[0],residuumFull[1]}  -- store for next iteration
  c.rA = residuumFull[2] -- store for next iteration

  var p  = {c.pre[0] * c.r[0], c.pre[1] * c.r[1]} -- apply preconditioner M^-1
  c.p = p

  var pA = c.preA * c.rA -- apply preconditioner M^-1
  c.pA = pA

  scanAlphaNumerator += dot(c.r, p) + c.rA * pA
end

local ebb PCGStep1(c : grid.cells)
  var tmp : L.vec3f = applyJTJ(c) -- A x p_k  => J^T x J x p_k 
  c.Ap_X  = {tmp[0], tmp[1]}  -- store for next kernel call
  c.Ap_XA  = tmp[2] -- store for next kernel call
  scanAlphaDenominator += dot(c.p, {tmp[0], tmp[1]} ) + c.pA * tmp[2]  -- x-th term of denominator of alpha
end

local ebb PCGStep2(c : grid.cells)
  var alpha = scanAlphaNumerator/scanAlphaDenominator 
  c.delta = c.delta+alpha*c.p -- do a descent step
  c.deltaA = c.deltaA+alpha*c.pA
  
  var r = c.r-alpha*c.Ap_X -- update residuum
  c.r = r  -- store for next kernel call
  
  var rA = c.rA-alpha*c.Ap_XA -- update residuum
  c.rA = rA  -- store for next kernel call

  var z = mulV2(c.pre,r)  -- apply pre-conditioner M^-1
  c.z = z  -- save for next kernel call

  var zA = c.preA*rA  -- apply pre-conditioner M^-1
  c.zA = zA  -- save for next kernel call

  scanBetaNumerator += dot(z,r) +  zA * rA -- compute x-th term of the numerator of beta
end

local ebb PCGStep3(c : grid.cells)
  var beta = scanBetaNumerator/scanAlphaNumerator
  c.p  = c.z+beta*c.p  -- update descent direction
  c.pA = c.zA+beta*c.pA
end

local ebb PCGLinearUpdate(c : grid.cells)
  c.X = c.X + c.delta
  c.A = c.A + c.deltaA
end


local ebb measure_cost( c : grid.cells )
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
computeCost()
print( 'initial cost: ', tostring(cost:get()*0.5) )

for nIter=0, nIterations do
  scanAlphaNumerator:set(0)
  scanAlphaDenominator:set(0)
  scanBetaNumerator:set(0)
  grid.cells:foreach(PCGInit1)
  for lIter = 0, lIterations do       
    scanAlphaDenominator:set(0)           
    grid.cells:foreach(PCGStep1)
    scanBetaNumerator:set(0)
    grid.cells:foreach(PCGStep2)
    grid.cells:foreach(PCGStep3)
    -- save new rDotz for next iteration
    scanAlphaNumerator:set(scanBetaNumerator:get())
  end
  grid.cells:foreach(PCGLinearUpdate)
  computeCost()
  print( 'iteration #'..tostring(nIter), 'cost: ', tostring(cost:get()*0.5) )
end



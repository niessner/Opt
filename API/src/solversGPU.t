
local S = require("std")
local util = require("util")
local C = util.C
local Timer = util.Timer
local positionForValidLane = util.positionForValidLane

local gpuMath = util.gpuMath

solversGPU = {}


solversGPU.gradientDescentGPU = require("solverGPUGradientDescent")
solversGPU.gaussNewtonGPU = require("solverGPUGaussNewton")
return solversGPU
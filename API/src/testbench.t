local opt = require("o")

local bigt = setmetatable({}, { __index = function(idx) return 3 end })

opt.dimensions = bigt
opt.elemsizes = bigt
opt.strides = bigt
opt.problemkind = ""

opt.math = require("util").cpuMath
package.terrapath = package.terrapath..";../../Examples/SmoothingLaplacianFloat4/?.t;../testMLib/?.t;"
require("SmoothingLaplacianFloat4AD")


local opt = require("o")
opt.dimensions = {[0] = 1, 2, 3 }
opt.elemsizes = {[0] = 4,4,4}
opt.strides = {[0] = 4,4,4}
opt.math = require("util").cpuMath
package.terrapath = package.terrapath..";../testMLib/?.t"
require("imageSmoothingAD")

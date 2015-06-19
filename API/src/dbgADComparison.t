local S = require("std")
local opt = require("o")
local util = require("util")
local dbg = require("dbg")
C = util.C
local stb = terralib.includecstring [[
#include "stb_perlin.c"
]]


local W = 16
local H = 16


opt.dimensions = {}
opt.dimensions[0] = W--Wtype
opt.dimensions[1] = H--Htype

opt.elemsizes = {}
opt.strides = {}
for i=0,2 do 
   opt.elemsizes[i] = 4
   opt.strides[i] = 4*W
end
opt.problemkind = ""

local Wtype = opt.Dim("W", 0)
local Htype = opt.Dim("H", 1)



opt.math = require("util").cpuMath
local function makeSumResult(resultType)
   local terra sumResult(result : resultType)
      var sum = 0.0f
      for j=0,H do
	 for i=0,W do
	    sum = sum + C.fabsf(result(i,j))--result(i,j))
	 end
      end
      return sum
   end
   return sumResult
end


local function makeCompareCosts(p1,p2, imageStructType)
   local terra compareCost(images : imageStructType, params1 : p1:ParameterType(), params2 : p2:ParameterType())
      for j=0,H do
	 for i=0,W do
	    var cost1 = p1.functions.cost.boundary(i,j,i,j,params1)
	    var cost2 = p2.functions.cost.boundary(i,j,i,j,params2)
	    images.result1(i,j) = cost1
	    images.result2(i,j) = cost2
	    images.diff(i,j) = cost1 - cost2
	 end
      end
   end
   return compareCost
end

local function makeCompareJTFs(p1,p2, imageStructType)
   local unknownElement = p1:UnknownType().metamethods.typ
   local terra compareJTF(gImages : imageStructType, preImages : imageStructType, params1 : p1:ParameterType(), params2 : p2:ParameterType())
      for j=0,H do
	 for i=0,W do
	    var residuum1 : unknownElement 
	    var residuum2 : unknownElement 
	    var pre1 : unknownElement 
	    var pre2 : unknownElement 

	    residuum1, pre1 = p1.functions.evalJTF.boundary(i,j,i,j,params1)
	    residuum2, pre2 = p2.functions.evalJTF.boundary(i,j,i,j,params2)

	    gImages.result1(i,j) = residuum1
	    gImages.result2(i,j) = residuum2
	    
	    preImages.result1(i,j) = pre1
	    preImages.result2(i,j) = pre2
	    
	    gImages.diff(i,j) = residuum1 - residuum2
	    preImages.diff(i,j) = pre1 - pre2
	 end
      end
   end
   return compareJTF
end

local function makeCompareJTJs(ps1,ps2, imageStructType)
   local unknownElement = ps1:UnknownType().metamethods.typ
   local terra compareJTJ(images : imageStructType, params1 : ps1:ParameterType(), params2 : ps2:ParameterType())
      var p1 : ps1:UnknownType()
      p1:initCPU()
      var p2 : ps2:UnknownType()
      p2:initCPU()

      for j=0,H do
	 for i=0,W do
	    var residuum1 : unknownElement 
	    var residuum2 : unknownElement 
	    var pre1 : unknownElement 
	    var pre2 : unknownElement 

	    residuum1, pre1 = ps1.functions.evalJTF.boundary(i,j,i,j,params1)
	    residuum2, pre2 = ps2.functions.evalJTF.boundary(i,j,i,j,params2)
	    p1(i,j) = residuum1*pre1
	    p2(i,j) = residuum2*pre2
	 end
      end
      for j=0,H do
	 for i=0,W do
	    var cost1 = ps1.functions.applyJTJ.boundary(i,j,i,j,params1, p1)
	    var cost2 = ps2.functions.applyJTJ.boundary(i,j,i,j,params2, p2)
	    images.result1(i,j) = cost1
	    images.result2(i,j) = cost2
	    images.diff(i,j) = cost1 - cost2
	 end
      end
   end
   return compareJTJ
end

-- Operates on one channel floating point images
local terra perlinNoise(im : &float, width : int, height : int, seed : float)
   for j=0,height do
      for i=0,width do
	 var result = stb.stb_perlin_noise3([float](i)/width, [float](j)/height, seed, 0, 0, 0)
	 im[j*width+i] = result
      end
   end
end 

local function run(file1,file2)
   local p1 = opt.problemSpecFromFile(file1)
   local p2 = opt.problemSpecFromFile(file2)
   
   local sumResult = makeSumResult(p1:UnknownType())
   local unknownImage = p1.parameters[p1.names["X"]]

   local resultImageType = opt.newImage(float,Wtype,Htype,4,4*W) --TODO: need to pass type

   local struct diffImageData {
      result1 : resultImageType
      result2 : resultImageType
      diff    : resultImageType
   }
   terra diffImageData:initCPU()
      self.result1:initCPU()
      self.result2:initCPU()
      self.diff:initCPU()
   end

   local compareCosts = makeCompareCosts(p1,p2, diffImageData)
   local compareJTFs = makeCompareJTFs(p1, p2, diffImageData)
   local compareJTJs = makeCompareJTJs(p1, p2, diffImageData)

   local params = {}

   local terra printSumError(name : rawstring, im : resultImageType)
      var sum = sumResult(im)
      C.printf("%s Difference: %f\n", name, sum)
     
   end

   local terra printPercentDifference(name : rawstring, diffImages : diffImageData)
      var sum = sumResult(diffImages.diff)
      var costSum1 = sumResult(diffImages.result1)
      var costSum2 = sumResult(diffImages.result2)
      C.printf("%f - %f = %f\n", costSum1, costSum2, sum);
      C.printf("%s Difference: %f%%\n", name, 2.0f*100.0f*sum/(costSum1+costSum2))
   end

   local terra runComparison()
      var costImages : diffImageData
      var gradientImages : diffImageData
      var preconditionerImages : diffImageData
      var jtjImages : diffImageData

      costImages:initCPU()
      gradientImages:initCPU()
      preconditionerImages:initCPU()
      jtjImages:initCPU()

      -- TODO figure out how many images they are, what type they are, and generate based on that
      var images = [&&uint8](C.malloc(16)) 
      images[0] = [&uint8](C.malloc(W*H*4)) 
      C.memset(images[0], 0, W*H*4)
      images[1] = [&uint8](C.malloc(W*H*4)) 
      C.memset(images[1], 0, W*H*4)      
      perlinNoise([&float](images[0]), W, H, 0.0)
      perlinNoise([&float](images[1]), W, H, 1.0)

      var params1 = [util.getParameters(p1, images, nil, params)] -- TODO: need to pass parameters
      var params2 = [util.getParameters(p2, images, nil, params)] -- TODO: need to pass parameters

      compareCosts(costImages, params1, params2)
      compareJTFs(gradientImages, preconditionerImages, params1, params2)
      compareJTJs(jtjImages,  params1, params2)

      printPercentDifference("Cost", costImages)
      printPercentDifference("Gradient", gradientImages)
      printPercentDifference("Preconditioner", preconditionerImages)
      printPercentDifference("JTJ",               jtjImages)

   end
   runComparison()

end

run("../../API/testMLib/imageSmoothing.t",
    "../../API/testMLib/imageSmoothingAD.t")


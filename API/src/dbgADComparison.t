local S = require("std")
local opt = require("o")
local util = require("util")
local dbg = require("dbg")
local im = require("im")
C = util.C
local stb = terralib.includecstring [[
#include "stb_perlin.c"
]]


local W = 16
local H = 16

local terra absReduce(f : float)
   return C.fabsf(f)
end

local terra absReduce(v : opt.float2)
   return v:abs():sum()
end

local terra absReduce(v : opt.float3)
   return v:abs():sum()
end

local function makeSumResult(resultType)
   local terra sumResult(result : resultType)
      var sum : float = 0.0f
      for j=0,H do
	 for i=0,W do
	    sum = sum + absReduce(result(i,j))--result(i,j))
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
   print ("makeCompareJTJs ps1:UnknownType(): " .. tostring(ps1:UnknownType()))
   print ("makeCompareJTJs ps2:UnknownType(): " .. tostring(ps2:UnknownType()))
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

-- Operates on floating point images
local terra perlinNoise(im : &float, width : int, height : int, channels : int, seed : float)
   for j=0,height do
      for i=0,width do
	 for c=0,channels do
	    var result = stb.stb_perlin_noise3([float](i)/width, [float](j)/height, seed+c, 0, 0, 0)
	    -- add one to go positive
	    im[(j*width+i)*channels + c] = result + 1.0
	 end
      end
   end
end 


function imagedumpHeader(filename)
   local terra readHeader()
      var fileHandle = C.fopen(filename, 'rb')
      var width : int, height : int, numChannels :int, datatype : int
      C.fread(&width, sizeof(int), 1, fileHandle)
      C.fread(&height, sizeof(int), 1, fileHandle)
      C.fread(&numChannels, sizeof(int), 1, fileHandle)
      C.fread(&datatype, sizeof(int), 1, fileHandle)
      C.fclose(fileHandle)
      return {width,height,numChannels,datatype}
   end
   local header = readHeader()
   return header._0, header._1, header._2, header._3
end

-- see if the file exists
function fileExists(file)
   local f = io.open(file, "rb")
   if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function linesFrom(file)
   if not fileExists(file) then return {} end
   local lines = {}
   for line in io.lines(file) do 
      lines[#lines + 1] = line
  end
  return lines
end

local function loadParameters(filename)
   local numbers = {}
   local lineArray = linesFrom(filename)
--   print(lineArray[1])
   for k,line in ipairs(lineArray) do
      local n = tonumber(line)
      print(n)
  --    print(line)
      numbers[#numbers + 1] = n

   end
   return numbers
end

function tableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

local function run(file1, file2, inputImageTypesFile, inputImageFiles, inputParameterFile)
   opt.dimensions = {}

   opt.elemsizes = {}
   opt.strides = {}

   local doLoadImages = false
   local imFilenames = {}
   print(inputImageFiles[1])
   if #inputImageFiles > 0 then
      doLoadImages = true
      imFilenames = inputImageFiles--linesFrom(inputImageFiles)
      local chans = {}
      local datatype = 0
      W, H, chans[1], datatype = imagedumpHeader(imFilenames[1])
      opt.elemsizes[0] = 4*chans[1]
      opt.strides[0] = 4*W*chans[1]
      for i=2,#imFilenames do
	 local newWidth = 0
	 local newHeight = 0
	 newWidth, newHeight, chans[i], datatype = imagedumpHeader(imFilenames[i])
	 assert(W == newWidth and H == newHeight, 
		"Widths and heights of images not identical")
	 opt.elemsizes[i-1] = 4*chans[i]
	 opt.strides[i-1] = 4*W*chans[i]
      end
   else 
      local chans = loadParameters(inputImageTypesFile)

      for i=0,#chans-1 do 
	 opt.elemsizes[i] = 4*chans[i+1]
	 opt.strides[i] = 4*W*chans[i+1]
      end
   end

   opt.dimensions[0] = W--Wtype
   opt.dimensions[1] = H--Htype


   opt.problemkind = ""
   
   local Wtype = opt.Dim("W", 0)
   local Htype = opt.Dim("H", 1)

   opt.math = require("util").cpuMath

   local p1 = opt.problemSpecFromFile(file1)
   local p2 = opt.problemSpecFromFile(file2)
   
   local sumResult = makeSumResult(p1:UnknownType())
   local unknownImage = p1.parameters[p1.names["X"]]

   local imageTypes = {}
   local paramTypes = {}

   for key, entry in ipairs(p1.parameters) do
      if entry.kind == "image" then
	 imageTypes[entry.idx] = entry.type
	 print("" .. key .. " = imageTypes[" .. entry.idx .. "] = " .. tostring(entry.type))
      elseif entry.kind == "param" then
	 paramTypes[entry.idx] = entry.type
      end
   end

   local p2ParamCount = 0
   local p2ImageCount = 0
   for _, entry in ipairs(p2.parameters) do
      if entry.kind == "image" then
	 assert(imageTypes[entry.idx], 
		"Image " .. entry.idx .. " exists only in second problem spec")
	 assert(imageTypes[entry.idx] == entry.type, 
		"Image type mismatch (" .. entry.idx .. "): " .. tostring(imageTypes[entry.idx]) .. " != " .. tostring(entry.type))	    
	 p2ImageCount = p2ImageCount + 1
      elseif entry.kind == "param" then
	 assert(paramTypes[entry.idx], 
		"Parameter " .. entry.idx .. " exists only in second problem spec")
	 assert(paramTypes[entry.idx] == entry.type, 
		"Param type mismatch (" .. entry.idx .. "): " .. tostring(paramTypes[entry.idx]) .. " != " .. tostring(entry.type))
	 p2ParamCount = p2ParamCount + 1
      end
   end
   -- TODO: print what parameter(s) missing
   assert(tableLength(paramTypes) == p2ParamCount, 
	  "Parameter count mismatch: " .. tableLength(paramTypes) .. " vs " .. p2ParamCount)
   assert(tableLength(imageTypes) == p2ImageCount, 
	  "Image count mismatch: " .. tableLength(imageTypes) .. " vs " .. p2ImageCount)


   local resultImageType = p1.parameters[p1.names["X"]].type
   print("Result Image Type: " .. tostring(resultImageType))
   local imCount = tableLength(imageTypes)
   local strides = terralib.new(int[imCount])
   local channels = terralib.new(int[imCount])
   for i = 0,imCount-1 do
      strides[i] = imageTypes[i].metamethods.stride

      local pixelType = imageTypes[i].metamethods.typ
      if pixelType == float then
	 channels[i] = 1 
      elseif pixelType == opt.float2 then
	 channels[i] = 2
      elseif pixelType == opt.float3 then
	 channels[i] = 3
      elseif pixelType == opt.float4 then
	 channels[i] = 4
      else
	 assert(false, "Only floating point image types are available in the test bench. Pixel type found: " .. tostring(pixelType))
      end
   end

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

   local params_lua = {}
   local paramData = nil
   if inputParameterFile then
      params_lua = loadParameters(inputParameterFile)
      paramData = terralib.new(float[#params_lua])
      for i = 1,#params_lua do
	 paramData[i-1] = params_lua[i]
      end
   end

   assert(#params_lua == p2ParamCount, "The parameter file has a different amount of parameters than is required by the terra files: ".. #params_lua .. " vs " .. p2ParamCount)


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

      var images : &&uint8
      if [doLoadImages] then
	 images = [&&uint8](C.malloc([tableLength(imFilenames)]*sizeof([&&uint8]))) 
	 var width : int, height : int, numChannels : int
	 escape 
	    for i = 0,tableLength(imFilenames)-1 do
	       emit `im.imageRead([&&float](&images[i]), &width, &height, &numChannels, [imFilenames[i+1]]) 
	    end
	 end
      else
	 images = [&&uint8](C.malloc([tableLength(imageTypes)]*sizeof([&&uint8]))) 
	 for i = 0,[tableLength(imageTypes)] do
	    var imSize = strides[i]*H
	    images[i] = [&uint8](C.malloc(imSize))
	    C.memset(images[i], 0, imSize)
	    if i < 3 then
	       perlinNoise([&float](images[i]), W, H, channels[i], [float](i))
	    end
	    --[[var buffer : int8[128]
	    C.sprintf(buffer, "image%d.imagedump", i)
	       im.imageWrite([&float](images[i]), W, H, channels[i], buffer)]]--
	 end	 
      end

      var finalParams : &&opaque = nil
      var paramData : &float
      
      if [not not inputParameterFile] then
	 finalParams = [&&opaque](C.malloc([#params_lua]*8))
	 for i = 0,[#params_lua-1] do
	    finalParams[i] = &paramData[i]
	 end
      end

      var params1 = [util.getParameters(p1, images, nil, finalParams)] -- TODO: need to pass parameters
      var params2 = [util.getParameters(p2, images, nil, finalParams)] -- TODO: need to pass parameters

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
local argparse = require "argparse"

local parser = argparse("dbgADComparison.t", "Compares two Opt problems to each other. Optionally specify the input images (an image types), and input parameters. If none are specified, we provide randomly generated defaults.")
local arg = parser:argument("input0", "First terra input file.")
arg:default("../testMLib/imageSmoothing.t")
arg = parser:argument("input1", "Second terra input file.")
arg:default("../testMLib/imageSmoothingAD.t")
local option = parser:option("-i --images", "Image Files"):count("*")
option = parser:option("-p --parameters", "Parameter File")
option = parser:option("-t --imagetypes", "File with # of floating-point channels for each image, 1 per line")
--parser:option("-I --include", "Include locations."):count("*")
local args = parser:parse()




if not (args["imagetypes"] or (#args["images"] > 0)) then
    print("Must have at least one of inputImageTypesFile and inputImageFile") 
    print(parser:get_help())
else
    run(args["input0"], args["input1"], args["imagetypes"], args["images"], args["parameters"])
end




local util = require("util")
local C = util.C
local dbg = require("dbg")
local im = require("im")
local stb = terralib.includecstring [[
#include "stb_perlin.c"
]]


-- Operates on floating point images
local terra perlinNoise(img : &float, width : int, height : int, channels : int, seed : float)
   for j=0,height do
      for i=0,width do
	 for c=0,channels do
	    var result = stb.stb_perlin_noise3([float](i)/width, [float](j)/height, seed+c, 0, 0, 0)
	    img[(j*width+i)*channels + c] = result
	 end
      end
   end
end 



terra imageTest(width : int, height : int, channels : int) 
   C.printf("imageTestBegin\n")
   var imPtr : &float
   im.initImage(&imPtr, width, height, channels)
   C.printf("imageTestInitImage\n")
   perlinNoise(imPtr, width, height, channels, 0.0f)
   C.printf("imageTestPerlin\n")
   var filename = "testImage.imagedump"
   im.imageWrite(imPtr, width, height, channels, filename)
   C.printf("imageTestWrite\n")
   var imPtr2 : &float

   var width2 : int, height2 : int, channels2 : int
   im.imageRead(&imPtr2, &width2, &height2, &channels2, filename)
   C.printf("imageTestRead\n")
   for j=0,height do
      for i=0,width do
	 for c=0,channels do
	    dbg.assert(imPtr[(j*width+i)*channels + c] == imPtr2[(j*width+i)*channels + c], "Pixels not the same")
	 end
      end
   end
   C.free(imPtr)
   C.free(imPtr2)
end
imageTest(16, 16, 3)


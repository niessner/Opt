local util = require("util")
local C = util.C
local im = require("im")
local dbg = {}

terra dbg.initCudaImage(imPtr : &&float, width : int, height : int, numChannels : int)
   var numBytes : int = sizeof(float)*width*height*numChannels
   C.printf("Num bytes: %d\n", numBytes)
   var err = C.cudaMalloc([&&opaque](imPtr), numBytes)
   if err ~= 0 then C.printf("cudaMalloc error: %d", err) end
end


terra dbg.imageWriteFromCuda(imPtr : &float, width : int, height : int, channelCount : int, filename : rawstring)
   var size = sizeof(float) * [uint32](width*height*channelCount)
   var ptr = C.malloc(size)
   C.cudaMemcpy(ptr, imPtr, size, C.cudaMemcpyDeviceToHost)
   im.imageWrite(ptr, width, height, channelCount, filename)
   C.free(ptr)
end

terra dbg.imageWriteFromCudaPrefix(imPtr : &float, width : int, height : int, channelCount : int, prefix : rawstring)
   var buffer : int8[128]
   --var suffix = "AD"
   var suffix = "optNoAD"
   C.sprintf(buffer, "%s_%s.imagedump", prefix, suffix)
   dbg.imageWriteFromCuda(imPtr, width, height, channelCount, buffer)
end

terra dbg.assert(condition : bool, msg : rawstring)
   if not condition then
      C.printf("%s\n", msg)
      C.exit(1)
   end
end

return dbg

local util = require("util")
local C = util.C
local dbg = {}

terra dbg.initImage(imPtr : &&float, width : int, height : int, numChannels : int)
   var numBytes : int = sizeof(float)*width*height*numChannels
   C.printf("Num bytes: %d\n", numBytes)
   var err = C.cudaMalloc([&&opaque](imPtr), numBytes)
   if err ~= 0 then C.printf("cudaMalloc error: %d", err) end
end

terra dbg.imageWrite(imPtr : &float, width : int, height : int, channelCount : int, filename : rawstring)
   var datatype : int = 0 -- floating point
   var fileHandle = C.fopen(filename, 'wb') -- b for binary
   C.fwrite(&width, sizeof(int), 1, fileHandle)
   C.fwrite(&height, sizeof(int), 1, fileHandle)
   C.fwrite(&channelCount, sizeof(int), 1, fileHandle)
   C.fwrite(&datatype, sizeof(int), 1, fileHandle)
   
   var size = sizeof(float) * [uint32](width*height*channelCount)
   var ptr = C.malloc(size)
   C.cudaMemcpy(ptr, imPtr, size, C.cudaMemcpyDeviceToHost)
   C.fwrite(ptr, sizeof(float), [uint32](width*height*channelCount), fileHandle)
   C.fclose(fileHandle)
   
   C.free(ptr)
   
end

terra dbg.imageWritePrefix(imPtr : &float, width : int, height : int, channelCount : int, prefix : rawstring)
   var buffer : int8[128]
   --var suffix = "AD"
   var suffix = "optNoAD"
   C.sprintf(buffer, "%s_%s.imagedump", prefix, suffix)
   dbg.imageWrite(imPtr, width, height, channelCount, buffer)
end

return dbg

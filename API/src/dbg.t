local util = require("util")
local C = util.C
local dbg = {}

terra dbg.initCudaImage(imPtr : &&float, width : int, height : int, numChannels : int)
   var numBytes : int = sizeof(float)*width*height*numChannels
   C.printf("Num bytes: %d\n", numBytes)
   var err = C.cudaMalloc([&&opaque](imPtr), numBytes)
   if err ~= 0 then C.printf("cudaMalloc error: %d", err) end
end

-- Up to the user to free the pointer in imPtr
terra dbg.initImage(imPtr : &&float, width : int, height : int, numChannels : int)
   var numBytes : int = sizeof(float)*width*height*numChannels
   @imPtr = [&float](C.malloc(numBytes))
end

terra dbg.imageWrite(imPtr : &float, width : int, height : int, channelCount : int, filename : rawstring)
   var datatype : int = 0 -- floating point
   var fileHandle = C.fopen(filename, 'wb') -- b for binary
   C.fwrite(&width, sizeof(int), 1, fileHandle)
   C.fwrite(&height, sizeof(int), 1, fileHandle)
   C.fwrite(&channelCount, sizeof(int), 1, fileHandle)
   C.fwrite(&datatype, sizeof(int), 1, fileHandle)
   
   C.fwrite(imPtr, sizeof(float), [uint32](width*height*channelCount), fileHandle)
   C.fclose(fileHandle)
end

terra dbg.imageWriteFromCuda(imPtr : &float, width : int, height : int, channelCount : int, filename : rawstring)
   var size = sizeof(float) * [uint32](width*height*channelCount)
   var ptr = C.malloc(size)
   C.cudaMemcpy(ptr, imPtr, size, C.cudaMemcpyDeviceToHost)
   dbg.imageWrite(ptr, width, height, channelCount, filename)
   C.free(ptr)
end

terra dbg.imageWriteFromCudaPrefix(imPtr : &float, width : int, height : int, channelCount : int, prefix : rawstring)
   var buffer : int8[128]
   --var suffix = "AD"
   var suffix = "optNoAD"
   C.sprintf(buffer, "%s_%s.imagedump", prefix, suffix)
   dbg.imageWriteFromCuda(imPtr, width, height, channelCount, buffer)
end

-- mallocs the ptr to the image data, and stores the ptr in the imPtr handle
terra dbg.imageRead(imPtr : &&float, width : &int, height : &int, numChannels : &int, filename : rawstring)
   var fileHandle = C.fopen(filename, 'rb')
   var datatype : int = 0
   C.fread(width, sizeof(int), 1, fileHandle)
   C.fread(height, sizeof(int), 1, fileHandle)
   C.fread(numChannels, sizeof(int), 1, fileHandle)
   C.fread(&datatype, sizeof(int), 1, fileHandle)
   --TODO: assert
   @imPtr = [&float](C.malloc(sizeof(float)*@width*@height*@numChannels))
   C.fread(@imPtr, sizeof(float), [uint32](@width*@height*@numChannels), fileHandle)
   C.fclose(fileHandle)
end

terra dbg.assert(condition : bool, msg : rawstring)
   if not condition then
      C.printf("%s\n", msg)
      C.exit(1)
   end
end

return dbg

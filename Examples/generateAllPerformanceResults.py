from subprocess import call

import os

#TODO: Add directory as parameter

#TODO: Error Handling

# [Example Name, Executable name, arg0, arg1, arg2]


def perfTest(exampleName, exeName, arg0, arg1, arg2):
	os.chdir("./" + exampleName)
	print("PerfTest " + exampleName)
	call(["x64/Release/" + exeName + ".exe", arg0, arg1, arg2])
	print("Done with " + exampleName)
	os.chdir("../")

def scalingPerfTest():
	os.chdir("./ImageWarping")
	print("*****************Scaling Tests.***********************")
	for i in range(0,6):
	    call(["x64/Release/ImageWarping.exe", str(2**i), "perf"])
	os.chdir("../")


testTable = {}
testTable["ImageWarping"] = ("ImageWarping", "ImageWarping", "0", "perf", "")
testTable["MeshDeformationARAP"] = ("MeshDeformationARAP", "MeshDeformation", "Armadillo20k.ply", "perf", "")
testTable["ShapeFromShading"] = ("ShapeFromShadingSimple", "ShapeFromShadingSimple", "default", "perf", "")
testTable["CotangentLaplacian"] = ("MeshSmoothingLaplacianCOT", "MeshSmoothingLaplacianCOT", "serapis.stl", "perf", "")

performanceTests = [testTable["ShapeFromShading"], testTable["CotangentLaplacian"], testTable["ImageWarping"], testTable["MeshDeformationARAP"]]

for pTest in performanceTests:
	perfTest(*pTest)

scalingPerfTest()





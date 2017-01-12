from subprocess import call

from utils import *
from opt_utils import *

import os
import platform
import sys

s = platform.system()
windows = (s == 'Windows') or (s == 'Microsoft')
freebsd = (s == 'FreeBSD')
linux   = (s == 'Linux')
osx     = (s == 'Darwin')

#TODO: Add directory as parameter

#TODO: Error Handling

# [Example Name, Executable name, arg0, arg1, arg2]

# TODO: handle windows/unix differences
def compileProject(exampleName, exampleProjectName):
	os.chdir(examples_dir + exampleName)
	if windows:
		VisualStudio(exampleProjectName + ".sln", [exampleProjectName])
	else:
		call(["make"])
	os.chdir(script_dir)


def perfTest(exampleName, exeName, arg0, arg1, arg2):
	os.chdir(examples_dir + exampleName)
	make_sure_path_exists("results/")
	print("PerfTest " + exampleName)
	call(["x64/Release/" + exeName + ".exe", arg0, arg1, arg2])
	print("Done with " + exampleName)
	os.chdir(script_dir)

def scalingPerfTest():
	os.chdir(examples_dir + "ImageWarping")
	print("*****************Scaling Tests.***********************")
	for i in range(0,6):
	    call(["x64/Release/ImageWarping.exe", str(2**i), "perf"])
	os.chdir(script_dir)


def buildAndRunPerformanceTests(pTests):
	for pTest in performanceTests:
		compileProject(pTest[0], pTest[0])
		perfTest(*pTest)
	scalingPerfTest()



testTable = {}
testTable["ImageWarping"] = ("ImageWarping", "ImageWarping", "0", "perf", "")
testTable["MeshDeformationARAP"] = ("MeshDeformationARAP", "MeshDeformationARAP", "Armadillo20k.ply", "perf", "")
testTable["MeshDeformationLARAP"] = ("MeshDeformationLARAP", "MeshDeformationLARAP", "meshes/neptune.ply", "", "")
testTable["ShapeFromShading"] = ("ShapeFromShadingSimple", "ShapeFromShadingSimple", "default", "perf", "")
testTable["CotangentLaplacian"] = ("MeshSmoothingLaplacianCOT", "MeshSmoothingLaplacianCOT", "serapis.stl", "perf", "")
#  testTable["CotangentLaplacian"], 
performanceTests = [testTable["MeshDeformationLARAP"], testTable["MeshDeformationARAP"], testTable["ImageWarping"], testTable["ShapeFromShading"]]

pascalOrBetterGPU = False
if len(sys.argv) > 1 and "true" in sys.argv[1]:
	pascalOrBetterGPU = True
	print("Enabling fast double precision atomic add")

os.chdir(example_dir)
setExcludeEnabled(True)
setContiguousAllocation(False)
setUtilParams(False, pascalOrBetterGPU)
setCusparseParams(False, False)


setDoublePrecision(False)
buildAndRunPerformanceTests(performanceTests)

setDoublePrecision(True)
buildAndRunPerformanceTests(performanceTests)






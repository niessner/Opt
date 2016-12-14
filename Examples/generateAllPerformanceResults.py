from subprocess import call

from utils import *

import os
import platform

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
	os.chdir("./" + exampleName)
	if windows:
		VisualStudio(exampleProjectName + ".sln", [exampleProjectName])
	else:
		call(["make"])
	os.chdir("../")


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


def buildAndRunPerformanceTests(pTests):
	for pTest in performanceTests:
		compileProject(pTest[0], pTest[0])
		perfTest(*pTest)
	scalingPerfTest()

def setDoublePrecision(isDouble):
	boolExpr = "false"
	intExpr = "0"
	typeString = "float"

	if isDouble:
		boolExpr = "true"
		intExpr = "1"
		typeString = "double"
	with open("shared/opt_precision.t", "w") as text_file:
		text_file.write("OPT_DOUBLE_PRECISION =  %s" % boolExpr)

	with open('shared/Precision.h.templ', 'r') as text_file:
		headerString = text_file.read()
	headerString = headerString.replace("$0", intExpr)

	with open("shared/Precision.h", "w") as text_file:
		text_file.write(headerString)

	optAPIprecisionString = """-- Switch to double to check for precision issues in the solver
-- currently using doubles is extremely slow
opt_float = """

	with open("../API/src/precision.t", "w") as text_file:
		text_file.write("%s %s" % (optAPIprecisionString, typeString))

testTable = {}
testTable["ImageWarping"] = ("ImageWarping", "ImageWarping", "0", "perf", "")
testTable["MeshDeformationARAP"] = ("MeshDeformationARAP", "MeshDeformationARAP", "Armadillo20k.ply", "perf", "")
testTable["ShapeFromShading"] = ("ShapeFromShadingSimple", "ShapeFromShadingSimple", "default", "perf", "")
testTable["CotangentLaplacian"] = ("MeshSmoothingLaplacianCOT", "MeshSmoothingLaplacianCOT", "serapis.stl", "perf", "")
#  testTable["CotangentLaplacian"], 
performanceTests = [testTable["MeshDeformationARAP"], testTable["ImageWarping"], testTable["ShapeFromShading"]]



setDoublePrecision(True)
buildAndRunPerformanceTests(performanceTests)

setDoublePrecision(False)
buildAndRunPerformanceTests(performanceTests)






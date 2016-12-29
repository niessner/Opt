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


def perfTest(exampleName, exeName, arg0="", arg1="", arg2=""):
	os.chdir("./" + exampleName)
	print("PerfTest " + exampleName)
	if arg0+arg1+arg2 == "":
		call(["x64/Release/" + exeName + ".exe"])
	else:
		call(["x64/Release/" + exeName + ".exe", arg0, arg1, arg2])
	print("Done with " + exampleName)
	os.chdir("../")

def buildAndRunPerformanceTests(pTests):
	for pTest in pTests:
		compileProject(pTest[0], pTest[0])
		perfTest(*pTest)

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

   #TODO: Add in Poisson Image Editing
projectList = ["ImageWarping", "IntrinsicLP", "MeshDeformationARAP", "MeshDeformationED", "MeshDeformationLARAP", 
    "MeshSmoothingLaplacian", "MeshSmoothingLaplacianCOT", "OpticalFlow", "RobustMeshDeformationARAP",
    "ShapeFromShadingSimple", "SmoothingLaplacianLP"]


for name in projectList:
    testTable[name] = (name, name)

costTests = [testTable[name] for name in projectList]

#setDoublePrecision(True)
buildAndRunPerformanceTests(costTests)






from subprocess import call

from utils import *
from opt_utils import *
import sys

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
	os.chdir(examples_dir + exampleName)
	if windows:
		VisualStudio(exampleProjectName + ".sln", [exampleProjectName])
	else:
		call(["make"])
	os.chdir(script_dir)


def perfTest(exampleName, exeName, arg0="", arg1="", arg2=""):
	os.chdir(examples_dir + exampleName)
	print("PerfTest " + exampleName)
	if arg0+arg1+arg2 == "":
		call(["x64/Release/" + exeName + ".exe"])
	else:
		call(["x64/Release/" + exeName + ".exe", arg0, arg1, arg2])
	print("Done with " + exampleName)
	os.chdir(script_dir)

def buildAndRunPerformanceTests(pTests):
	for pTest in pTests:
		compileProject(pTest[0], pTest[0])
		perfTest(*pTest)

testTable = {}

projectList = ["ImageWarping", "IntrinsicLP", "MeshDeformationARAP", "MeshDeformationED", "MeshDeformationLARAP", 
    "MeshSmoothingLaplacianCOT", "OpticalFlow", "RobustMeshDeformationARAP",
    "ShapeFromShadingSimple"]

cusparseMode = len(sys.argv) > 1 and ("false" in sys.argv[1])

if not cusparseMode:
    projectList.append("PoissonImageEditing")
    setCusparseParams(False, False)

for name in projectList:
    testTable[name] = (name, name)

costTests = [testTable[name] for name in projectList]

setExcludeEnabled(not cusparseMode)
setContiguousAllocation(cusparseMode)
setDoublePrecision(False)
setUtilParams(True, False)
buildAndRunPerformanceTests(costTests)






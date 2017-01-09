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

testTable = {}

projectList = ["ImageWarping", "IntrinsicLP", "MeshDeformationARAP", "MeshDeformationED", "MeshDeformationLARAP", 
    "MeshSmoothingLaplacianCOT", "OpticalFlow", "RobustMeshDeformationARAP",
    "ShapeFromShadingSimple"]


for name in projectList:
    testTable[name] = (name, name)

costTests = [testTable[name] for name in projectList]

#setDoublePrecision(True)
buildAndRunPerformanceTests(costTests)






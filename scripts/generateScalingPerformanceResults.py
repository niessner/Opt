from subprocess import call

from utils import *
from opt_utils import *
import shutil

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

def scalingPerfTest():
	compileProject("ImageWarping", "ImageWarping")
	os.chdir(examples_dir + "ImageWarping")
	print("*****************Scaling Tests.***********************")
	for i in range(0,6):
	    call(["x64/Release/ImageWarping.exe", str(2**i), "perf"])
	os.chdir(script_dir)


def copyResultsToResultDirectory(pTests):
	make_sure_path_exists(results_dir)
	for p in pTests:
		copytree(examples_dir + p[0] + "/results/", results_dir + p[0])

pascalOrBetterGPU = False
if len(sys.argv) > 1 and "true" in sys.argv[1]:
	pascalOrBetterGPU = True
	print("Enabling fast double precision atomic add")

setQTolerance(1e-9)
setExcludeEnabled(True)
setContiguousAllocation(False)
setUtilParams(False, pascalOrBetterGPU)
setCusparseParams(False, False)


setDoublePrecision(False)
scalingPerfTest()

setDoublePrecision(True)
scalingPerfTest()

copyResultsToResultDirectory([["ImageWarping"]])






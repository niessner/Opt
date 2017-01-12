from subprocess import call

from utils import *


def runPaperExamples(outputFile):
	with open(outputFile, "w") as outfile:
		subprocess.call(["python", "./runPaperExamples.py", "false"], stdout=outfile)

def setCusparseParams(usingCusparse, fusedJtJ):
	cusparseString = "local use_cusparse  = "
	fusedJtJString = "local use_fused_jtj = "
	if usingCusparse:
		cusparseString += "true\n"
	else:
		cusparseString += "false\n"
	if fusedJtJ:
		fusedJtJString += "true\n"
	else:
		fusedJtJString += "false\n"

	solveFilename = '../API/src/solverGPUGaussNewton.t'

	solverLines = []
	with open(solveFilename, 'r') as text_file:
		solverLines = text_file.readlines()
	for i in range(len(solverLines)):
		if "local use_cusparse" in solverLines[i]:
			solverLines[i] = cusparseString
			solverLines[i+1] = fusedJtJString
			break
	with open(solveFilename, 'w') as text_file:
		text_file.write("".join(solverLines))


def runCusparsePaperExamples(name, usingCusparse, fusedJtJ):
	print("Generating "+name+" results")
	setCusparseParams(usingCusparse, fusedJtJ)
	runPaperExamples("cusparseTiming_" + name + ".txt")

runCusparsePaperExamples("noCusparse", False, False)
runCusparsePaperExamples("cusparseSeparateJTJ", True, False)
runCusparsePaperExamples("cusparseCombinedJTJ", True, True)
print("Finished")





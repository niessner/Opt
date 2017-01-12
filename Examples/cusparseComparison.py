from subprocess import call

from utils import *
from opt_utils import *


def runPaperExamples(outputFile):
	with open(outputFile, "w") as outfile:
		subprocess.call(["python", "./runPaperExamples.py", "false"], stdout=outfile)

def runCusparsePaperExamples(name, usingCusparse, fusedJtJ):
	print("Generating "+name+" results")
	setCusparseParams(usingCusparse, fusedJtJ)
	runPaperExamples("cusparseTiming_" + name + ".txt")

runCusparsePaperExamples("noCusparse", False, False)
runCusparsePaperExamples("cusparseSeparateJTJ", True, False)
runCusparsePaperExamples("cusparseCombinedJTJ", True, True)
print("Finished")





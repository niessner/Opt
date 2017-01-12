from subprocess import call

from utils import *
from opt_utils import *


def runPaperExamples(outputFile):
	with open(outputFile, "w") as outfile:
		subprocess.call(["python", script_dir + "runPaperExamples.py", "false"], stdout=outfile)

def runCusparsePaperExamples(name, usingCusparse, fusedJtJ):
	print("Generating "+name+" results")
	setCusparseParams(usingCusparse, fusedJtJ)
	runPaperExamples(results_dir + "cusparseTiming_" + name + ".txt")

ensure_result_dirs()
runCusparsePaperExamples("noCusparse", False, False)
runCusparsePaperExamples("cusparseSeparateJTJ", True, False)
runCusparsePaperExamples("cusparseCombinedJTJ", True, True)
print("Finished")





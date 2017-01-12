import errno
import os

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

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


def setUtilParams(timeIndividualKernels, pascalOrBetterGPU):
	timeIndividualKernelsString = "local timeIndividualKernels  = "
	pascalOrBetterGPUString = "local pascalOrBetterGPU = "
	if timeIndividualKernels:
		timeIndividualKernelsString += "true\n"
	else:
		timeIndividualKernelsString += "false\n"
	if pascalOrBetterGPU:
		pascalOrBetterGPUString += "true\n"
	else:
		pascalOrBetterGPUString += "false\n"

	utilFilename = '../API/src/util.t'

	utilLines = []
	with open(utilFilename, 'r') as text_file:
		utilLines = text_file.readlines()
	for i in range(len(utilLines)):
		if "local timeIndividualKernels" in utilLines[i]:
			utilLines[i] = timeIndividualKernelsString
			utilLines[i+1] = pascalOrBetterGPUString
			break
	with open(utilFilename, 'w') as text_file:
		text_file.write("".join(utilLines))

def setContiguousAllocation(contiguousAllocation):
	boolExpr = "false"
	if contiguousAllocation:
		boolExpr = "true"
	oFilename = '../API/src/o.t'
	oLines = []
	with open(oFilename, 'r') as text_file:
		oLines = text_file.readlines()
	for i in range(len(oLines)):
		if "local use_contiguous_allocation" in oLines[i]:
			oLines[i] = "local use_contiguous_allocation = "+boolExpr+"\n"
			break
	with open(oFilename, 'w') as text_file:
		text_file.write("".join(oLines))

def setExcludeEnabled(excludeEnabled):
	filenames = ['ShapeFromShadingSimple/shapeFromShadingAD.t', 'ImageWarping/imageWarpingAD.t']
	for fName in filenames:
		lines = []
		with open(fName, 'r') as text_file:
			lines = text_file.readlines()
		for i in range(len(lines)):
			if "Exclude" in lines[i]:
				if excludeEnabled:
					lines[i] = lines[i].replace("-","")
				else:
					lines[i] = "--" + lines[i]
				break
		with open(fName, 'w') as text_file:
			text_file.write("".join(lines))
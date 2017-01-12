import errno
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = script_dir + "../results"
opt_src_dir = script_dir + "../API/src/"
examples_dir = script_dir + "../Examples/"
simple_dir = examples_dir + "SimpleCERESComparison/"
simple_results_dir = results_dir + "simple/"

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def ensure_result_dirs():
	make_sure_path_exists(results_dir)
	make_sure_path_exists(simple_results_dir)


def setLocalLuaBooleansInFile(filename, var_names, boolExprs):
	lines = []
	with open(filename, 'r') as text_file:
		lines = text_file.readlines()
	for v,b in zip(var_names, boolExprs):
		baseExpr = "local " + v
		boolString = "false\n"
		if b:
			boolString = "true\n"
		for i in range(len(lines)):
			if baseExpr in lines[i]:
				lines[i] = baseExpr + boolString
				break
	with open(filename, 'w') as text_file:
		text_file.write("".join(lines))


def setCusparseParams(usingCusparse, fusedJtJ):
	filename = opt_src_dir + 'solverGPUGaussNewton.t'
	var_names = ["use_cusparse", "use_fused_jtj"]
	boolExprs = [usingCusparse, fusedJtJ]
	setLocalLuaBooleansInFile(filename, var_names, boolExprs)

def setUtilParams(timeIndividualKernels, pascalOrBetterGPU):
	filename = opt_src_dir + 'util.t'
	var_names = ["timeIndividualKernels", "pascalOrBetterGPU"]
	boolExprs = [timeIndividualKernels, pascalOrBetterGPU]
	setLocalLuaBooleansInFile(filename, var_names, boolExprs)

def setContiguousAllocation(contiguousAllocation):
	filename = opt_src_dir + 'o.t'
	setLocalLuaBooleansInFile(filename, ["use_contiguous_allocation"], [contiguousAllocation])

def setDoublePrecision(isDouble):
	boolExpr = "false"
	intExpr = "0"
	typeString = "float"

	if isDouble:
		boolExpr = "true"
		intExpr = "1"
		typeString = "double"
	with open(examples_dir + "shared/opt_precision.t", "w") as text_file:
		text_file.write("OPT_DOUBLE_PRECISION =  %s" % boolExpr)

	with open(examples_dir + 'shared/Precision.h.templ', 'r') as text_file:
		headerString = text_file.read()
	headerString = headerString.replace("$0", intExpr)

	with open(examples_dir + "shared/Precision.h", "w") as text_file:
		text_file.write(headerString)

	optAPIprecisionString = """-- Switch to double to check for precision issues in the solver
-- currently using doubles is extremely slow
opt_float = """

	with open(opt_src_dir + "precision.t", "w") as text_file:
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

	utilFilename = opt_src_dir + 'util.t'

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
	oFilename = opt_src_dir + 'o.t'
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
	filenames = [examples_dir + 'ShapeFromShadingSimple/shapeFromShadingAD.t', examples_dir + 'ImageWarping/imageWarpingAD.t']
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

import errno
import os
import shutil

script_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
results_dir = script_dir + "../results/"
opt_src_dir = script_dir + "../API/src/"
examples_dir = script_dir + "../Examples/"
simple_dir = examples_dir + "SimpleCERESComparison/"
simple_results_dir = results_dir + "simple/"


def thousandAsKFormatter(x,p):
	return str(int(x/1000)) + "k"

# CERES, optLM_d, optGN_d, optLM_f, optGN_f
graph_colors = ['#608ac3', '#2ca02c', '#d62728', '#9467bd', '#FD8B3A']

#For graphs
LEGEND_SIZE = 12
SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 28

def copytree(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def ensure_result_dirs():
	make_sure_path_exists(results_dir)
	make_sure_path_exists(simple_results_dir)

def setVariablesInFile(filename, decl_strings, assignment_strings):
	lines = []
	with open(filename, 'r') as text_file:
		lines = text_file.readlines()
	for d,a in zip(decl_strings, assignment_strings):
		for i in range(len(lines)):
			if d in lines[i]:
				lines[i] = d + " = " + a
				break
	with open(filename, 'w') as text_file:
		text_file.write("".join(lines))

def setLocalLuaBooleansInFile(filename, var_names, boolExprs):
	decl_strings = ["local "+ v for v in var_names]
	assignment_strings = ["true\n" if b else "false\n" for b in boolExprs]
	setVariablesInFile(filename, decl_strings, assignment_strings)

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

def setQTolerance(q_tolerance):
	filename = opt_src_dir + 'solverGPUGaussNewton.t'
	setVariablesInFile(filename, ["        var q_tolerance"], [str(q_tolerance)+"\n"])

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

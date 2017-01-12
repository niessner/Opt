from os import listdir
from os.path import isfile, join
from opt_utils import *

originalPath = simple_dir + "originalDataSheets/"
dataPath = simple_dir + "data/"
dataSheets = [f for f in listdir(originalPath) if isfile(join(originalPath, f)) and ".dat" in f]

ceresStructStubs = []
ceresSelectorStubs = []

def makeCeresStruct(name, codeSnippet, paramCount):
	ceresCode = "struct Term" + name + "\n{\n\t\tTerm" + name
	ceresCode += """(double x, double y) : x(x), y(y) {}
	template <typename T>
	bool operator()(const T* const funcParams, T* residuals) const
	{
		/* """
	ceresCode += codeSnippet
	ceresCode += " */\n"
	for i in range(0,paramCount):
		ceresCode += "\t\tT b" + str(i+1) + " = funcParams[" + str(i) +"];\n"
	ceresCode += "\t\tresiduals[0] = " + codeSnippet.replace("=", "-")[:-6]
	ceresCode += """;\n\t\treturn true;
	}
	static ceres::CostFunction* Create(double x, double y)
	{
		return (new ceres::AutoDiffCostFunction<Term""" + name + ", 1, " + str(paramCount) + """>(
			new Term""" + name + """(x, y)));
	}
	double x, y;
};\n\n"""
	return ceresCode

def makeCeresSelector(name):
	return 'if (problemInfo.baseName == "' + name.lower() + '") costFunction = Term' + name + "::Create(functionData[i].x, functionData[i].y);"

for fname in dataSheets:
	with open(join(originalPath, fname)) as f:
		sheet = f.readlines()
		count = 0
		dataStartLine = -1
		for idx, line in enumerate(sheet):
			if "Data:" in line:
				count += 1
				if count > 1:
					dataStartLine = idx + 1
					break
		cleanData = [" ".join([str(float(x)) for x in line.split()]) for line in sheet[dataStartLine:]]
		resultContents = "\n".join(cleanData)
		name = fname[:-4]
		outputfname = name.lower() + ".txt"

		valuestart = [i for i,line in enumerate(sheet) if "Start 1" in line][0] + 1
		start1 = []
		start2 = []
		solution = []
		for line in sheet[valuestart:]:
			tokens = line.split()
			if len(tokens) == 0 or not tokens[0].startswith("b"):
				break
			start1.append(float(tokens[2]))
			start2.append(float(tokens[3]))
			solution.append(float(tokens[4]))

		paramCount = len(start1)
		cppCode = '\tproblems.push_back(NLLSProblem("' + name.lower()
		cppCode += '", ' + str(paramCount) + ", { "
		cppCode += ", ".join([str(x) for x in start1])
		cppCode += " }, { "
		cppCode += ", ".join([str(x) for x in solution])
		cppCode += "}));"

		codeStart = [i for i,line in enumerate(sheet) if "Model:" in line][0] + 3
		codeSnippet = sheet[codeStart].lstrip()
		for line in sheet[codeStart+1:]:
			if "b" not in line:
				break
			codeSnippet += line
		codeSnippet = codeSnippet.rstrip()
		ceresStructStubs.append(makeCeresStruct(name, codeSnippet, paramCount))
		ceresSelectorStubs.append(makeCeresSelector(name))

		print(cppCode)
		#print(outputfname)
		with open(join(dataPath, outputfname), "w") as outputFile:
			outputFile.write(resultContents)
print("".join(ceresStructStubs))
print("\n".join(ceresSelectorStubs))
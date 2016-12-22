from os import listdir
from os.path import isfile, join

originalPath = "originalDataSheets/"
dataPath = "data/"
dataSheets = [f for f in listdir(originalPath) if isfile(join(originalPath, f)) and ".dat" in f]

ceresStructStubs = []

def makeCeresStruct(name, codeSnippet, paramCount):
	ceresCode = "struct Term" + name + "\n{\nTerm" + name
	ceresCode += """(double x, double y) : x(x), y(y) {}
	template <typename T>
	bool operator()(const T* const funcParams, T* residuals) const
	{
		/*\n"""
	ceresCode += codeSnippet
	ceresCode += "\t\t*/\n"
	for i in range(0,paramCount):
		ceresCode += "\t\tT b" + str(i+1) + " = funcParams[" + str(i) +"];\n"
	ceresCode += "residuals[0] = " + codeSnippet
	ceresCode += """\t\treturn true;
	}
	static ceres::CostFunction* Create(double x, double y)
	{
		return (new ceres::AutoDiffCostFunction<Term""" + name + ", 1, " + str(paramCount) + """>(
			new Term""" + name + """(x, y)));
	}
	double x, y;
};"""
	return ceresCode

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

		ceresStructStubs.append(makeCeresStruct(name, paramCount, codeSnippet))

		print(cppCode)
		#print(outputfname)
		with open(join(dataPath, outputfname), "w") as outputFile:
			outputFile.write(resultContents)
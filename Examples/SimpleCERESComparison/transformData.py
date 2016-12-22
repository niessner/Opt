from os import listdir
from os.path import isfile, join

originalPath = "originalDataSheets/"
dataPath = "data/"
dataSheets = [f for f in listdir(originalPath) if isfile(join(originalPath, f)) and ".dat" in f]

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
		name = fname[:-4].lower()
		outputfname = name + ".txt"

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

		cppCode = '\tproblems.push_back(NLLSProblem("' + name
		cppCode += '", ' + str(len(start1)) + ", { "
		cppCode += ", ".join([str(x) for x in start1])
		cppCode += " }, { "
		cppCode += ", ".join([str(x) for x in solution])
		cppCode += "}));"
		print(cppCode)
		#print(outputfname)
		with open(join(dataPath, outputfname), "w") as outputFile:
			outputFile.write(resultContents)
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
		outputfname = fname[:-4].lower() + ".txt"
		#print(outputfname)
		with open(join(dataPath, outputfname), "w") as outputFile:
			outputFile.write(resultContents)
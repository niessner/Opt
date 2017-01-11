from os import listdir
from os.path import isfile, join

originalPath = "originalDataSheets/"
dataSheets = [f for f in listdir(originalPath) if isfile(join(originalPath, f)) and ".dat" in f]

problemNames = [filename[:-4] for filename in dataSheets]

trueSolutions = []
ceresStatus = []
doubleLMStatus = []
floatLMStatus = []
optCERESEnergiesMatch = []
optCERESIterationsMatch = []
difficulties = []

def approxEqual(x, y, epsilon=0.005):
	return abs(x-y)/max(abs(x), abs(y)) < epsilon

def nearIdentical(trueSolution, otherSolution, epsilon=0.005):
	return all([approxEqual(x,y) for x,y in zip(trueSolution, otherSolution)])

lowerDifficulty = {"Misra1a", "Chwirut2", "Chwirut1", "Lanczos3", "Gauss1", "Gauss2", "DanWood", "Misra1b"}
averageDifficulty = {"Kirby2", "Hahn1", "Nelson", "MGH17", "Lanczos1", "Lanczos2", "Gauss3", "Misra1c", "Misra1d", "Roszman1", "ENSO"}
higherDifficulty = {"MGH09", "Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4", "Rat43", "Bennett5"}

def toStatus(b):
	return "Good" if b else "Bad"

for name in problemNames:
	if name.lower() == "nelson": # Until we make it work
		continue

	doubleRows = []
	floatRows = []

	with open(join("results/", name.lower() + "_double.csv")) as f:
		doubleRows = f.readlines()
	with open(join("results/", name.lower() + "_float.csv")) as f:
		floatRows = f.readlines()

	trueSolRow = doubleRows[1].split(",")
	ceresSolRow = doubleRows[2].split(",")
	doubleLMSolRow = doubleRows[3].split(",")
	floatLMSolRow = floatRows[3].split(",")
	firstIterRow = doubleRows[6].split(",")
	trueSolution 	= [float(x) for x in trueSolRow[2:]]
	ceresSolution 	= [float(x) for x in ceresSolRow[2:]]
	doubleLMSolution 	= [float(x) for x in doubleLMSolRow[2:]]
	floatLMSolution 	= [float(x) for x in floatLMSolRow[2:]]
	ceresOK = nearIdentical(trueSolution, ceresSolution)
	doubleLMOK = nearIdentical(trueSolution, doubleLMSolution)
	floatLMOK = nearIdentical(trueSolution, floatLMSolution)
	ceresStatus.append(ceresOK)
	doubleLMStatus.append(doubleLMOK)
	floatLMStatus.append(floatLMOK)
	optCERESEnergiesMatch.append(approxEqual(float(firstIterRow[1]), float(firstIterRow[2])))
	
	iterationsMatch = True
	for row in doubleRows[6:]:
		cells = row.split(",")
		if len(cells) > 2 and not approxEqual(float(cells[1]), float(cells[2])):
				iterationsMatch = False
				break
	difficulty = "Higher" if name in higherDifficulty else ("Average" if name in averageDifficulty else ("Lower" if name in lowerDifficulty else "Unknown"))
	difficulties.append(difficulty)
	optCERESIterationsMatch.append(iterationsMatch)

output = "," + ", ".join([name.lower() for name in problemNames if not name.lower() == "nelson"]) + "\n"
output += "Difficulty," + ",".join(difficulties) + "\n"
output += "Energy Match?," + ",".join([toStatus(s) for s in optCERESEnergiesMatch]) + "\n"
output += "Iterations Match?," + ",".join([toStatus(s) for s in optCERESIterationsMatch]) + "\n"
output += "CERES," + ",".join([toStatus(s) for s in ceresStatus]) + "\n"
output += "OptLM (double)," + ",".join([toStatus(s) for s in doubleLMStatus]) + "\n"
output += "OptLM (float)," + ",".join([toStatus(s) for s in floatLMStatus]) + "\n"

with open(join("results/", "summary.csv"), "w") as f:
	f.write(output)
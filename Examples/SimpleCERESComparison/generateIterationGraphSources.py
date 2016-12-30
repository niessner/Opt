
summary = []
with open("results/summary.csv", "r") as text_file:
	summary = text_file.readlines()
problems = summary[0].split(",")
indicators = summary[3].split(",")
badProblems = []

for i,v in enumerate(indicators):
	if v == "Bad":
		badProblems.append(problems[i].strip())
print(badProblems)

for p in badProblems:
	filename = "results/"+p+"_double.csv"
	lines = []
	with open(filename, "r") as text_file:
		lines = text_file.readlines()[4:]
	splitlines = [line.split(",") for line in lines]
	newlines = [line[0] + "," + line[1] + "," + line[2] for line in splitlines]
	outfile = "baditers/"+p+".csv"
	with open(outfile, "w") as text_file:
		text_file.write("\n".join(newlines))
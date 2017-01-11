import os
import errno
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

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

outdirectory = "iterations/"
make_sure_path_exists(outdirectory)

for p in [p.strip() for p in problems[1:]]:
	filename_d = "results/"+p+"_double.csv"
	lines_d = []
	with open(filename_d, "r") as text_file:
		lines_d = text_file.readlines()[6:]
	splitlines_d = [line.strip().split(",") for line in lines_d]

	filename_f = "results/"+p+"_float.csv"
	lines_f = []
	with open(filename_f, "r") as text_file:
		lines_f = text_file.readlines()[6:]
	splitlines_f = [line.strip().split(",") for line in lines_f]

	newlines = [d[0] + "," + d[1] + "," + d[2] + "," + d[3] + "," + f[2] + "," + f[3] for d,f in zip(splitlines_d,splitlines_f)]
	outfile = outdirectory+p+".csv"
	with open(outfile, "w") as text_file:
		text_file.write("\n".join(newlines))
import sys
from utils import run
import glob
logs = glob.glob('./*.log')
for filename in logs:
	print(filename)
	with open(filename, 'r', encoding='ascii') as text_file:
		lines = text_file.readlines()
		lines = [L for L in lines if not L.isspace()]
	indices = [i for i,v in enumerate(lines) if "===" in v]
	for i in indices:
		if lines[i+1][:2] == "**":
			print("".join(lines[i:i+4]))
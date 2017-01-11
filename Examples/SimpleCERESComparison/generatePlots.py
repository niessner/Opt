import numpy as np
import matplotlib.pyplot as plt
import os
import errno
plt.style.use('ggplot')
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

plotpath = "plots/"
make_sure_path_exists(plotpath)

summary = []
with open("results/summary.csv", "r") as text_file:
	summary = text_file.readlines()
problems = [p.strip() for p in summary[0].split(",")][1:]

def processData(col):
	data = []
	for i,v in enumerate(col):
		if "QNAN" in v or "INF" in v or "IND" in v:
			data.append(data[i-1])
		else:
			data.append(float(v))
	return data



for p in problems:
	filename = "iterations/"+p+".csv"
	lines = []
	with open(filename, "r") as text_file:
		lines = text_file.readlines()
	rows = [line.strip().split(",") for line in lines]
	iters = [r[0] for r in rows]
	ceres = processData([r[1] for r in rows])
	optLM_d = processData([r[2] for r in rows])
	optGN_d = processData([r[3] for r in rows])
	optLM_f = processData([r[4] for r in rows])
	optGN_f = processData([r[5] for r in rows])
	# red dashes, blue squares and green triangles

#iters, optLM_f, 'p', iters, optGN_f, 'y'
	plt.clf()
	ceres_line, = plt.plot(iters, ceres, '-g', linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor='g', label='CERES')
	optLM_d_line, = plt.plot(iters, optLM_d, '-b', marker=(3,0,0), markerfacecolor='none', markeredgecolor='b', label='Opt LM (double)')
	optGN_d_line, = plt.plot(iters, optGN_d, '-r', marker=(4,2,45), label='Opt GN (double)')
	optLM_f_line, = plt.plot(iters, optLM_f, '-c', marker=(6,2,0), label='Opt LM (float)')
	optGN_f_line, = plt.plot(iters, optGN_f, '-m', marker=(4,0,0), markerfacecolor='none', markeredgecolor='m', label='Opt GN (float)')
	plt.legend(handles=[ceres_line, optLM_d_line, optGN_d_line, optLM_f_line, optGN_f_line])
	plt.ylabel('Cost')
	plt.xlabel('Iterations')
	plt.title(p)
	#plt.plot(iters, optGN_f, 'g')
	plt.yscale('log')
	#plt.show()
	plt.savefig(plotpath+p+".pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

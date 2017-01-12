import numpy as np
import matplotlib.pyplot as plt
import os
import errno
from opt_utils import *

def processData(col):
	data = []
	for i,v in enumerate(col):
		if "QNAN" in v or "INF" in v or "IND" in v:
			data.append(data[i-1])
		else:
			data.append(float(v))
	return data

def make_iteration_plot(name, iters_f, iters_d, ceres, optLM_d, optGN_d, optLM_f, optGN_f):
	plt.style.use('ggplot')

	plt.clf()
	ceres_line, = plt.plot(iters_d, ceres, '-g', linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor='g', label='CERES')
	optLM_d_line, = plt.plot(iters_d, optLM_d, '-b', marker=(3,0,0), markerfacecolor='none', markeredgecolor='b', label='Opt LM (double)')
	optGN_d_line, = plt.plot(iters_d, optGN_d, '-r', marker=(4,2,45), label='Opt GN (double)')
	optLM_f_line, = plt.plot(iters_f, optLM_f, '-c', marker=(6,2,0), label='Opt LM (float)')
	optGN_f_line, = plt.plot(iters_f, optGN_f, '-m', marker=(4,0,0), markerfacecolor='none', markeredgecolor='m', label='Opt GN (float)')
	plt.legend(handles=[ceres_line, optLM_d_line, optGN_d_line, optLM_f_line, optGN_f_line])
	plt.ylabel('Cost')
	plt.xlabel('Iterations')
	SIZE = 12
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 20
	plt.title(p, fontsize=BIGGER_SIZE)
	plt.yscale('log')

	plt.rc('font', size=SIZE)                # controls default text sizes
	plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('legend', fontsize=SIZE)          # legend fontsize
	#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.show()
	plt.savefig(plotpath+p+".pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

def make_time_plot(name, ceres, optLM_d, optGN_d, optLM_f, optGN_f):
	plt.style.use('ggplot')

	plt.clf()
	ceres_line, = plt.plot(ceres[1], ceres[0], '-g', linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor='g', label='CERES')
	optLM_d_line, = plt.plot(optLM_d[1], optLM_d[0], '-b', marker=(3,0,0), markerfacecolor='none', markeredgecolor='b', label='Opt LM (double)')
	optGN_d_line, = plt.plot(optGN_d[1], optGN_d[0], '-r', marker=(4,2,45), label='Opt GN (double)')
	optLM_f_line, = plt.plot(optLM_f[1], optLM_f[0], '-c', marker=(6,2,0), label='Opt LM (float)')
	optGN_f_line, = plt.plot(optGN_f[1], optGN_f[0], '-m', marker=(4,0,0), markerfacecolor='none', markeredgecolor='m', label='Opt GN (float)')
	plt.legend(handles=[ceres_line, optLM_d_line, optGN_d_line, optLM_f_line, optGN_f_line])
	plt.ylabel('Cost')
	plt.xlabel('Iterations')
	SIZE = 12
	MEDIUM_SIZE = 16
	BIGGER_SIZE = 20
	plt.title(p, fontsize=BIGGER_SIZE)
	plt.yscale('log')

	plt.rc('font', size=SIZE)                # controls default text sizes
	plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('legend', fontsize=SIZE)          # legend fontsize
	#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.show()
	plt.savefig(plotpath+p+"_time.pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)


ensure_result_dirs()
plotpath = results_dir + "plots/"
make_sure_path_exists(plotpath)

problems = ["MeshDeformationLARAP", "MeshDeformationARAP", "ImageWarping", "ShapeFromShading"]
for problem in problems:
	suffix = "672.csv" if "ImageWarping" in problem else ".csv"
	file_prefix = results_dir + problem + "/results"
	rows_f = []
	rows_d = []
	with open(file_prefix+"float"+suffix, "r") as text_file:
		rows_f = text_file.readlines()[1:]
	with open(file_prefix+"double"+suffix, "r") as text_file:
		rows_d = text_file.readlines()[1:]
	iters_f = [r[0] for r in rows_f]
	iters_d = [r[0] for r in rows_d]
	#Iter, Ceres Error, Opt(GN) Error (double),  Opt(LM) Error (double), Ceres Iter Time(ms), Opt(GN) Iter Time(ms) (double), Opt(LM) Iter Time(ms) (double), Total Ceres Time(ms), Total Opt(GN) Time(ms) (double), Total Opt(LM) Time(ms) (double)
	errorCERES = processData([r[1] for r in rows_d])
	errorOptGN_d = processData([r[2] for r in rows_d])
	errorOptLM_d = processData([r[3] for r in rows_d])
	errorOptGN_f = processData([r[2] for r in rows_f])
	errorOptLM_f = processData([r[3] for r in rows_f])
	timeCERES = processData([r[7] for r in rows_d])
	timeOptGN_d = processData([r[8] for r in rows_d])
	timeOptLM_d = processData([r[9] for r in rows_d])
	timeOptGN_f = processData([r[8] for r in rows_f])
	timeOptLM_f = processData([r[9] for r in rows_f])
	make_iteration_plot(problem, iters_f, iters_d, errorCERES, errorOptLM_d, errorOptGN_d, errorOptLM_f, errorOptGN_f)
	ceres = zip(errorCERES, timeCERES)
	optLM_d = zip(errorOptLM_d, timeOptLM_d)
	optGN_d = zip(errorOptGN_d, timeOptGN_d)
	optLM_f = zip(errorOptLM_f, timeOptLM_f)
	optGN_f = zip(errorOptGN_f, timeOptGN_f)
	make_time_plot(problem, ceres, optLM_d, optGN_d, optLM_f, optGN_f)
	
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
	ceres_line, = plt.plot(iters_d[:len(ceres)], ceres, 	  color=graph_colors[0], linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor=graph_colors[0], label='CERES')
	optLM_d_line, = plt.plot(iters_d[:len(optLM_d)], optLM_d, color=graph_colors[1],  marker=(3,0,0), markerfacecolor='none', markeredgecolor=graph_colors[1], label='Opt LM (double)')
	optGN_d_line, = plt.plot(iters_d[:len(optGN_d)], optGN_d, color=graph_colors[2],  marker=(4,2,45), label='Opt GN (double)')
	optLM_f_line, = plt.plot(iters_f[:len(optLM_f)], optLM_f, color=graph_colors[3],  marker=(6,2,0), label='Opt LM (float)')
	optGN_f_line, = plt.plot(iters_f[:len(optGN_f)], optGN_f, color=graph_colors[4],  marker=(4,0,0), markerfacecolor='none', markeredgecolor=graph_colors[4], label='Opt GN (float)')


	iters = [iters_d, iters_d, iters_d, iters_f, iters_f]
	errors = [ceres, optLM_d, optGN_d, optLM_f, optGN_f]

	offsets = [0.0, 0.0, 0.0, 0.0, 0.0]
	#offsets = [-0.1, -0.2, -0.3, -0.4, -0.5]
	sizes = [2,2,2,2,2]

	#Auto-compute offsets
	maxIters = max([len(L) for L in errors])
	existingX = set()
	for i in range(len(errors)):
		xVal = iters[i][len(errors[i])-1]
		if xVal in existingX:
			continue
		existingX.add(xVal)
		count = 1
		for j in range(i+1,len(errors)):
			if (iters[j][len(errors[j])-1]) == xVal:
				count += 1
		if count > 1:
			offXEach = -0.002*maxIters
			offXTotal = offXEach
			offsets[i] = offXTotal
			offXTotal += offXEach
			for j in range(i+1,len(errors)):
				if (iters[j][len(errors[j])-1]) == xVal:
					sizes[j] = 1
					offsets[j] = offXTotal
					offXTotal += offXEach



	for i in range(5):
		plt.axvline(x=float(iters[i][len(errors[i])-1])+offsets[i], color=graph_colors[i], linestyle='--',    linewidth=sizes[i])

	plt.legend(handles=[ceres_line, optGN_f_line, optLM_f_line, optGN_d_line, optLM_d_line])
	plt.ylabel('Log Cost')
	plt.xlabel('Iterations')
	plt.title(name, fontsize=BIGGER_SIZE)
	plt.yscale('log',basey=2)

	plt.rc('font', size=SIZE)                # controls default text sizes
	plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('legend', fontsize=LEGEND_SIZE)          # legend fontsize
	#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.show()
	plt.tight_layout()
	plt.savefig(plotpath+name+".pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)

def make_time_plot(name, ceres, optLM_d, optGN_d, optLM_f, optGN_f):
	plt.style.use('ggplot')
	plt.clf()
	ceres_line, = plt.plot(*zip(*ceres), 	 color=graph_colors[0], linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor=graph_colors[0], label='CERES')
	optLM_d_line, = plt.plot(*zip(*optLM_d), color=graph_colors[1], marker=(3,0,0), markerfacecolor='none', markeredgecolor=graph_colors[1], label='Opt LM (double)')
	optGN_d_line, = plt.plot(*zip(*optGN_d), color=graph_colors[2], marker=(4,2,45), label='Opt GN (double)')
	optLM_f_line, = plt.plot(*zip(*optLM_f), color=graph_colors[3], marker=(6,2,0), label='Opt LM (float)')
	optGN_f_line, = plt.plot(*zip(*optGN_f), color=graph_colors[4], marker=(4,0,0), markerfacecolor='none', markeredgecolor=graph_colors[4], label='Opt GN (float)')

	plt.axvline(x=ceres[-1][0], color=graph_colors[0], linestyle='--',    linewidth=2)
	plt.axvline(x=optLM_d[-1][0], color=graph_colors[1], linestyle='--',  linewidth=2)
	plt.axvline(x=optGN_d[-1][0], color=graph_colors[2], linestyle='--',  linewidth=2)
	plt.axvline(x=optLM_f[-1][0], color=graph_colors[3], linestyle='--',  linewidth=2)
	plt.axvline(x=optGN_f[-1][0], color=graph_colors[4], linestyle='--',  linewidth=2)


	plt.legend(handles=[ceres_line, optGN_f_line, optLM_f_line, optGN_d_line, optLM_d_line])
	plt.ylabel('Log Cost')
	plt.xlabel('Log Time (ms)')
	plt.title(name, fontsize=BIGGER_SIZE)
	plt.xscale('log')
	plt.yscale('log',basey=2)

	plt.rc('font', size=SIZE)                # controls default text sizes
	plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('legend', fontsize=LEGEND_SIZE)          # legend fontsize
	#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.show()
	plt.tight_layout()
	plt.savefig(plotpath+name+"_time.pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)



def truncateErrors(errors, minError):
	for i, item in enumerate(errors):
		if item <= minError:
			return errors[:i+1]
	return errors

def scaleAll(listOfLists, scaleFactor):
	for errors in listOfLists:
		for i, item in enumerate(errors):
			errors[i] *= scaleFactor

ensure_result_dirs()
plotpath = results_dir + "plots/"
make_sure_path_exists(plotpath)

problems = ["MeshDeformationLARAP", "MeshDeformationARAP", "ImageWarping", "ShapeFromShadingSimple"]
for problem in problems:
	suffix = "672.csv" if "ImageWarping" in problem else ".csv"
	file_prefix = results_dir + problem + "/results_"
	rows_f = []
	rows_d = []
	with open(file_prefix+"float"+suffix, "r") as text_file:
		rows_f = [line.strip().split(",") for line in text_file.readlines()[1:]]
	with open(file_prefix+"double"+suffix, "r") as text_file:
		rows_d = [line.strip().split(",") for line in text_file.readlines()[1:]]
	iters_f = [r[0] for r in rows_f]
	iters_d = [r[0] for r in rows_d]
	#Iter, Ceres Error, Opt(GN) Error (double),  Opt(LM) Error (double), Ceres Iter Time(ms), Opt(GN) Iter Time(ms) (double), Opt(LM) Iter Time(ms) (double), Total Ceres Time(ms), Total Opt(GN) Time(ms) (double), Total Opt(LM) Time(ms) (double)
	errorCERES = processData([r[1] for r in rows_d])
	errorOptGN_d = processData([r[2] for r in rows_d])
	errorOptLM_d = processData([r[3] for r in rows_d])
	errorOptGN_f = processData([r[2] for r in rows_f])
	errorOptLM_f = processData([r[3] for r in rows_f])



	minError = errorCERES[-1]
	errorCERES = truncateErrors(errorCERES,minError)
	errorOptGN_d = truncateErrors(errorOptGN_d,minError)
	errorOptLM_d = truncateErrors(errorOptLM_d,minError)
	errorOptGN_f = truncateErrors(errorOptGN_f,minError)
	errorOptLM_f = truncateErrors(errorOptLM_f,minError)

	if problem == "ShapeFromShadingSimple":
		problem = "ShapeFromShading"
		#scaleAll([errorCERES,errorOptGN_d,errorOptLM_d,errorOptGN_f,errorOptLM_f], 2.5)

	startingRadius = 10000.0
	decreaseFactor = 2.0
	while errorCERES[0] == errorCERES[1]:
		errorCERES = errorCERES[1:]
		startingRadius /= decreaseFactor
		decreaseFactor *= 2.0
	if startingRadius < 10000.0:
		print(problem + ": started CERES at radius " + str(startingRadius))





	timeCERES = processData([r[7] for r in rows_d])
	timeOptGN_d = processData([r[8] for r in rows_d])
	timeOptLM_d = processData([r[9] for r in rows_d])
	timeOptGN_f = processData([r[8] for r in rows_f])
	timeOptLM_f = processData([r[9] for r in rows_f])
	make_iteration_plot(problem, iters_f, iters_d, errorCERES, errorOptLM_d, errorOptGN_d, errorOptLM_f, errorOptGN_f)
	ceres = list(zip(timeCERES, errorCERES))
	optLM_d = list(zip(timeOptLM_d, errorOptLM_d))
	optGN_d = list(zip(timeOptGN_d, errorOptGN_d))
	optLM_f = list(zip(timeOptLM_f, errorOptLM_f))
	optGN_f = list(zip(timeOptGN_f, errorOptGN_f))

	print(problem + ": Opt GN (float): %fms, %fx faster than CERES\n" % (timeOptGN_f[len(errorOptGN_f)-1], timeCERES[-1]/timeOptGN_f[len(errorOptGN_f)-1]))
	print(problem + ": Opt GN (double): %fms, %fx faster than CERES\n" % (timeOptGN_d[len(errorOptGN_d)-1], timeCERES[-1]/timeOptGN_d[len(errorOptGN_d)-1]))

	make_time_plot(problem, ceres, optLM_d, optGN_d, optLM_f, optGN_f)

i = 0
figureString=""
for p in problems:
	figureString += "\\includegraphics[width=0.5\\textwidth]{plots/" + p + ".pdf}"
	figureString += "\\includegraphics[width=0.5\\textwidth]{plots/" + p + "_time.pdf}"
	figureString += "\\\\\n"
print(figureString)
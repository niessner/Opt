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

def make_scaling_plot(name, scaling_sizes, ceres_pcg, ceres_cholesky, optLM_f, opt_LMd):
	plt.style.use('ggplot')
	plt.clf()
	ceres_line, = plt.plot(scaling_sizes, ceres_pcg, 	  color=graph_colors[0], linewidth=2, marker=(4,0,45), markerfacecolor='none', markeredgecolor=graph_colors[0], label='CERES (PCG)')
	optLM_d_line, = plt.plot(scaling_sizes, optLM_d, color=graph_colors[1],  marker=(3,0,0), markerfacecolor='none', markeredgecolor=graph_colors[1], label='Opt LM (double)')
	ceres_cholesky_line, = plt.plot(scaling_sizes, ceres_cholesky, color=graph_colors[2],  marker=(4,2,45), label='CERES (Cholesky)')
	optLM_f_line, = plt.plot(scaling_sizes, optLM_f, color=graph_colors[3],  marker=(6,2,0), label='Opt LM (float)')


	


	plt.rc('font', size=SIZE)                # controls default text sizes
	plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
	plt.rc('legend', fontsize=LEGEND_SIZE)          # legend fontsize

	plt.ylabel('Time to Convergence (ms, log scale)')
	plt.xlabel('# Unknowns (log scale)')
	plt.title(name, fontsize=BIGGER_SIZE)
	plt.yscale('log',basey=10)
	plt.xscale('log',basex=10)
	legend = plt.legend(handles=[ceres_line, ceres_cholesky_line, optLM_f_line, optLM_d_line],loc="upper left")

	ax = plt.gca()
	ax.set_axis_bgcolor('white')
	ax.yaxis.grid(color='#DDDDDD')
	ax.xaxis.grid(color='#DDDDDD')

	print(str(int(ceres_pcg[-1]/optLM_f[-1]))+"x")
	"""
	x = 0.97
	y = 0.725
	width = 0.05
	widthB = 3.4
	lengthB = 1.0

	plt.annotate(str(int(ceres_pcg[-1]/optLM_f[-1]))+"x", xy=(x,y), xytext=(x+width,y), xycoords="axes fraction", 
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB='+str(widthB)+', lengthB='+str(lengthB), lw=2.0,facecolor='black',edgecolor='black'))
	"""
	


	#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	#plt.show()
	plt.tight_layout()
	plt.savefig(plotpath+"ScalingIW.pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)


def truncateErrors(errors, minError):
	for i, item in enumerate(errors):
		if item <= minError:
			return errors[:i+1]
	return errors

ensure_result_dirs()
plotpath = results_dir + "plots/"
make_sure_path_exists(plotpath)

image_sizes = [1,2,4,8,16,32,64,128,256,512,1024]
scaling_sizes = []
ceres_pcg = [] 
ceres_cholesky = []
optLM_f = [] 
optLM_d = []
for image_size in image_sizes:
	suffix = str(image_size)+".csv"
	file_prefix = results_dir + "ImageWarping/results_"
	rows_f = []
	rows_d = []
	with open(file_prefix+"float"+suffix, "r") as text_file:
		rows_f = [line.strip().split(",") for line in text_file.readlines()[1:]]
	with open(file_prefix+"double"+suffix, "r") as text_file:
		rows_d = [line.strip().split(",") for line in text_file.readlines()[1:]]
	iters_f = [r[0] for r in rows_f]
	iters_d = [r[0] for r in rows_d]
	#Iter, Ceres Error, Opt(GN) Error (double),  Opt(LM) Error (double), Ceres Iter Time(ms), Opt(GN) Iter Time(ms) (double), Opt(LM) Iter Time(ms) (double), Total Ceres Time(ms), Total Opt(GN) Time(ms) (double), Total Opt(LM) Time(ms) (double)
	errorCERES_pcg = processData([r[1] for r in rows_d])
	errorOptLM_d = processData([r[3] for r in rows_d])
	errorOptLM_f = processData([r[3] for r in rows_f])
	errorCERES_cho = processData([r[1] for r in rows_f])


	minError = min(errorCERES_pcg[-1], errorCERES_cho[-1])
	if errorOptLM_d[-1] > minError:
		print("for size "+str(image_size)+" opt double cost not low enough")
	if errorOptLM_f[-1] > minError:
		print("for size "+str(image_size)+" opt float cost not low enough")

	errorCERES_pcg = truncateErrors(errorCERES_pcg,minError)
	errorCERES_cho = truncateErrors(errorCERES_cho,minError)
	errorOptLM_d = truncateErrors(errorOptLM_d,minError)
	errorOptLM_f = truncateErrors(errorOptLM_f,minError)

	timeCERES_pcg = processData([r[7] for r in rows_d])
	timeCERES_cho = processData([r[7] for r in rows_f])
	timeOptLM_d = processData([r[9] for r in rows_d])
	timeOptLM_f = processData([r[9] for r in rows_f])

	scaling_sizes.append(image_size*image_size*3)
	ceres_pcg.append(timeCERES_pcg[len(errorCERES_pcg)-1])
	ceres_cholesky.append(timeCERES_cho[len(errorCERES_cho)-1])
	optLM_f.append(timeOptLM_f[len(errorOptLM_d)-1])
	optLM_d.append(timeOptLM_d[len(errorOptLM_f)-1])
	
make_scaling_plot("Image Warping", scaling_sizes, ceres_pcg, ceres_cholesky, optLM_f, optLM_d)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import os
import errno
from opt_utils import *

plt.style.use('ggplot')

ensure_result_dirs()
plotpath = results_dir + "plots/"
make_sure_path_exists(plotpath)

#TODO: Pull automatically from somewhere
problems = ['Image Warping','Shape From Shading','ARAP Mesh Deformation','Volumetric Mesh Deformation','Cot. Mesh Smoothing','Optical Flow','Embedded Mesh Deformation','Intrinsic Images','Robust Non-Rigid\nAlignment']
matrixFreeLIterPerSecond = [14842.60347,6120.854471,2688.931819,24985.16506,3997.146038,6017.788583,2691.025108,3168.739928,7857.138367]
materializedLIterPerSecond = [4923.30188,2870.256795,1609.330483,13788.44391,13314.60412,3026.597741,2310.057807,1410.985937,5582.673392]
#materializedLIterPerSecond = [i*0.001 for i in materializedLIterPerSecond]
#matrixFreeLIterPerSecond = [i*0.001 for i in matrixFreeLIterPerSecond]

y_pos = np.arange(len(problems))
plt.clf()

LEGEND_SIZE=14
SMALL_SIZE=14
plt.rc('font', size=SIZE)                # controls default text sizes
plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)          # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)          # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)          # legend fontsize

TITLE_SIZE = 24

bar_width = 0.41

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FuncFormatter(thousandAsKFormatter))

matrixFreeBar = plt.barh(y_pos - bar_width*0.5, matrixFreeLIterPerSecond, bar_width, align='center', color=graph_colors[4])
materializedBar = plt.barh(y_pos + bar_width*0.5, materializedLIterPerSecond, bar_width, align='center', color=graph_colors[0])

ax.set_axis_bgcolor('white')
ax.yaxis.grid(color='#DDDDDD')
ax.xaxis.grid(color='#DDDDDD')

plt.legend([matrixFreeBar, materializedBar], ["Matrix-Free","Materialized JtJ"])
plt.yticks(y_pos, problems)
plt.xlabel('Linear Iterations Per Second')
#suptitle = plt.suptitle("Performance of Materialization Strategies", fontsize=TITLE_SIZE, x=0.32, y=1.06)
suptitle = plt.suptitle('Performance of Materialization Strategies', y=1.015, fontsize=TITLE_SIZE)



plt.tight_layout()
#plt.show()
plt.savefig(plotpath+"AllMaterialization.pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_extra_artists=(suptitle,), bbox_inches="tight", pad_inches=0.01,
        frameon=None)

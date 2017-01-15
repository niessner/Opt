import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import os
import errno
from matplotlib import ticker
from opt_utils import *

plt.style.use('ggplot')

ensure_result_dirs()
plotpath = results_dir + "plots/"
make_sure_path_exists(plotpath)

#reverse because matplotlib starts at the bottom
labels = ['None', 'Lighting', 'J', 'JtJ'][::-1]
y_pos = np.arange(len(labels))
linearItersPerSecond = [825.414613,6120.854471,1554.432097,2870.256795][::-1]
#linearItersPerSecond = [i*0.001 for i in linearItersPerSecond]
plt.clf()

TITLE_SIZE = 24

plt.rc('font', size=SIZE)                # controls default text sizes
plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)          # legend fontsize

font0 = FontProperties()
font0.set_size(TITLE_SIZE)
font0.set_weight('bold')

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FuncFormatter(thousandAsKFormatter))

plt.barh(y_pos, linearItersPerSecond, align='center', color=graph_colors[4])
plt.yticks(y_pos, labels)
plt.xlabel('Linear Iterations Per Second')
plt.title('Performance of Materialization Strategies', fontsize=TITLE_SIZE)
plt.annotate("Fastest", (3000, y_pos[2]-0.02), va='center', fontproperties=font0, color="black")



plt.tight_layout()
#plt.show()
plt.savefig(plotpath+"Materialization.pdf", dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)

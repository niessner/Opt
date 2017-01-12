from subprocess import call
from opt_utils import *
import os


os.chdir(examples_dir + "ImageWarping")
for i in range(0,6):
    call(["./x64/Release/ImageWarping.exe", str(2**i), "perf"])
os.chdir(script_dir)
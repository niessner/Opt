from subprocess import call

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


for i in range(0,6):
    call(["./x64/Release/ImageWarping.exe", str(2**i), "perf"])
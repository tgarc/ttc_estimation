#!/usr/bin/python
import matplotlib as mpl
import matplotlib.pyplot as plt 
import serial as ser
import signal, sys
import numpy as np
from functools import partial

def signal_handler(signal, frame):
    global ard
    print "\nCaught exception; closing serial port..."
    ard.close()
    sys.exit()


ard = ser.Serial("/dev/ttyACM0",9600,timeout=5)
ard.flushInput()
signal.signal(signal.SIGINT, signal_handler)    

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set_ylabel("Encoder1")
ax1.set_xlabel("Index")
lines = ax1.plot([], [], 'm-o', [], [], 'r-o')

redrawDelay = 5
bufSize = 10000

i = -1
encvals = np.zeros((2, bufSize), dtype=np.int64)
times = np.arange(bufSize, dtype=np.uint64)
for l in ard:
    i += 1

    # extend buffer size
    if((i % bufSize) == 0):
        encvals = np.concatenate((encvals, np.zeros_like(encvals)), axis=1)
        times = np.concatenate((times, np.zeros_like(times)), axis=1)

    vals = l.split(",")
    encvals[:, i] = map(int, vals)

    if ~(i % redrawDelay):
        [mpl.artist.setp(a, data=(times[:i+1], encvals[k, :i+1]))
         for k, a in enumerate(lines)]
        fig.canvas.draw()

    if ~(i % (redrawDelay*2)):
        ax1.relim()
        ax1.autoscale_view()

print "timed out."
ard.close()
plt.close()

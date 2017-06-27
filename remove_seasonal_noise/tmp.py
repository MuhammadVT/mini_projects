import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack
from read_data import read_data

# read the data
df = read_data("./original.mat")

## Number of samplepoints
#N = 600
## sample spacing
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#yf = scipy.fftpack.fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#fig1, ax1 = plt.subplots()
#ax1.plot(x, y)
#plt.show()

#fig, ax = plt.subplots()
#N = df.shape[0]     # number of data points
#xf = np.linspace(0, N/2, N/2)   # take one side frequency range
#yf = scipy.fftpack.fft(df.original) / N  # fft computing and normalization
#yf = np.abs(yf[:N/2]) # take one side  
#ax.plot(xf, yf)
#
#fig1, ax1 = plt.subplots()
#xx = np.arange(N)
#yy = 400 * np.sin(2*np.pi*xx/5.)
#ax1.plot(xx, yy)
#ax1.plot(xx, df.original)
#plt.show()

from seasonal import fit_seasons, adjust_seasons
fig, ax = plt.subplots()
#seasons, trend = fit_seasons(df.original, trend="mean")
seasons, trend = fit_seasons(df.original, trend="spline")
df['trend'] = trend
df.plot(ax=ax)
#ax.plot(seasons)
plt.show()



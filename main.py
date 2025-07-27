import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import toolkit as tk

# seed random numbers numpy
np.random.seed(0)

# initialize matplotlib parameters
plt.rcParams["figure.figsize"] = [7.50, 5.00]
plt.rcParams["figure.autolayout"] = True

# initialize sample rate
sample_rate = 3e9 #samples/sec

# initialize my target state
target = {
    "range" : 1000, #meters
    "rate" : -100, #m/s, negative closing, positive opening
    "RCS" : 10, # dBsm, Swerling 0
    }

# generate an array for signal
X,M = tk.chirped_waveform_single(1,sample_rate,1e9,10e6,100e-6,1e3)
pwr_X = (np.abs(M))
spec = 10*np.log10(np.abs(np.fft.fft(M)))

# plot the clean and noisy signals on top of each other
s = np.array(range(0,len(pwr_X)))
s = s/sample_rate
fig, axs = plt.subplots(2)
fig.suptitle('Signal Time and Freq Domain Analysis')
axs[0].set_title("Signal Amplitude")
axs[0].plot(s,pwr_X,'b')
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].yaxis.set_minor_locator(AutoMinorLocator())
axs[0].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
del s
# plot the clean and noisy signals on top of each other
s = np.array(range(0,len(spec)))
s = s*sample_rate/len(spec)
axs[1].set_title("FFT Output")
axs[1].plot(s,spec,'r')
axs[1].set_xlabel("Sampled Frequency")
axs[1].set_ylim([-20,80])
axs[1].set_ylabel("Power [dB]")
axs[1].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].yaxis.set_minor_locator(AutoMinorLocator())
plt.show()
del s
#################################### UNCLASSIFIED ####################################
# main.py
# Clayton Jaksha
# ALL RIGHTS RESERVED
# main script to conduct radar and EW modeling in a sandbox environment
# all work completed on personally owned devices with personally owned or licensed software


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import toolkit as tk
import scipy.constants

# seed random numbers numpy
np.random.seed(0)


# initialize matplotlib parameters
plt.rcParams["figure.figsize"] = [7.50, 5.00]
plt.rcParams["figure.autolayout"] = True

# initialize sample rate
sample_rate = 3e9 #samples/sec

# initialize my radar parameters
class radar:
    def __init__(self,P_t,G,L_s,T_s,BW_n):
        self.P_t = P_t # dBW, output power
        self.G = G # dBi, antenna gain
        self.L_s = L_s # dB, system losses (positive = loss)
        self.T_s = T_s # K, system temp
        self.BW_n = BW_n # Hz, system bandwidth tx/rx
        self.P_n = P_n = 10*np.log10(scipy.constants.k*T_s*BW_n) #default noise power, dB
rdr = radar(10*np.log10(1000),20,1,250,100e6)

# initialize my target state
class target:
    def __init__(self,range,rate,RCS):
        self.range = range #meters
        self.rate = rate #m/s, negative closing, positive opening
        self.RCS = RCS # dBsm, Swerling 0
tgt = target(5e3,-1000,10)

# generate an array for signal
X,M,f,amp = tk.chirped_waveform_single(np.sqrt(rdr.P_t),sample_rate,1e9,10e6,10e-6,10e3)
del amp
# apply some very light noise to the output (characterize a very high SNR on the TX function)
X_tx = tk.awgn(X,rdr.P_n,rdr.P_t)
pwr_X = 10*np.log10(np.multiply(X_tx,np.conjugate(X_tx)))
spec_X = 10*np.log10(np.abs(np.fft.fft(X_tx)))
# get the return pulse
Y = tk.return_pulse(tgt.range,tgt.rate,tgt.RCS,rdr.P_t,rdr.G,rdr.L_s,rdr.P_n,X_tx,f,10e-6,sample_rate)
pwr_Y = 10*np.log10(np.multiply(Y,np.conjugate(Y)))
spec_Y = 10*np.log10(np.abs(np.fft.fft(Y)))
# range process the pulse
fast_time = np.fft.fftshift(X_tx)


# plot the clean and noisy signals on top of each other
s_X = np.array(range(0,len(pwr_X)))
s_X = s_X/sample_rate
s_Y = np.array(range(0,len(pwr_Y)))
s_Y = s_Y/sample_rate
fig, axs = plt.subplots(2)
fig.suptitle('Signal Time and Freq Domain Analysis')
axs[0].set_title("Signal Power")
axs[0].plot(
    s_X,pwr_X,'b',
    s_Y,pwr_Y,'r')
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Signal Power [dBW]")
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].yaxis.set_minor_locator(AutoMinorLocator())
axs[0].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
del s_X,s_Y
# plot the clean and noisy signals on top of each other
s_X = np.array(range(0,len(spec_X)))
s_X = s_X*sample_rate/len(spec_X)
s_Y = np.array(range(0,len(spec_Y)))
s_Y = s_Y*sample_rate/len(spec_Y)
axs[1].set_title("FFT Output")
axs[1].plot(
    s_X,spec_X,'b',
    s_Y,spec_Y,'r')
axs[1].set_xlabel("Sampled Frequency")
#axs[1].set_ylim([-20,80])
axs[1].set_ylabel("Power [dB]")
axs[1].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].yaxis.set_minor_locator(AutoMinorLocator())
plt.show()
del s_X,s_Y






#################################### UNCLASSIFIED ####################################
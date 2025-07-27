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
rdr = radar(10*np.log10(100),20,1,300,100e6)

# initialize my target state
class target:
    def __init__(self,range,rate,RCS):
        self.range = range #meters
        self.rate = rate #m/s, negative closing, positive opening
        self.RCS = RCS # dBsm, Swerling 0
tgt = target(5e3,-100,10)

# generate an array for signal
X,M,f,amp = tk.chirped_waveform_single(np.sqrt(rdr.P_t),sample_rate,1e9,5e6,10e-6,10e3)
del amp
# apply some very light noise to the output (characterize a very high SNR on the TX function)
X_tx = tk.awgn(X,rdr.P_n,rdr.P_t)
# get the return pulse
Y = tk.return_pulse(tgt.range,tgt.rate,tgt.RCS,rdr.P_t,rdr.G,rdr.P_n,rdr.L_s,X_tx,f,0,sample_rate)
pwr_X = 10*np.log10(np.multiply(X_tx,np.conjugate(X_tx)))
spec = 10*np.log10(np.abs(np.fft.fft(X_tx)))

# plot the clean and noisy signals on top of each other
s = np.array(range(0,len(pwr_X)))
s = s/sample_rate
fig, axs = plt.subplots(2)
fig.suptitle('Signal Time and Freq Domain Analysis')
axs[0].set_title("Signal Power")
axs[0].plot(s,pwr_X,'b')
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Signal Power [dBW]")
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
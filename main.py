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
plt.rcParams["figure.figsize"] = [7.50, 6.00]
plt.rcParams["figure.autolayout"] = True

# initialize sample rate ENSURE THIS MEETS NYQUIST SAMPLING CRITERIA 2*f
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
rdr = radar(10*np.log10(5000),30,3,300,100e6)

# initialize my target state
class target:
    def __init__(self,range,rate,RCS):
        self.range = range #meters
        self.rate = rate #m/s, negative closing, positive opening
        self.RCS = RCS # dBsm, Swerling 0
tgt = target(25e3,-200,6)

# initialize waveform parameters
class waveform:
    def __init__(self,freq,BW,tau,PRF,CPI):
        self.freq = freq # Hz
        self.BW = BW # Hz
        self.tau = tau # s
        self.PRF = PRF # Hz
        self.CPI = CPI # s, should be a multiple of PRI
        self.PRI = 1/PRF # s
        self.wavelength = scipy.constants.c/freq # m
chirp = waveform(1e9,10e6,10e-6,10e3,100/10e3)

#m = int(chirp.CPI/chirp.PRI)
m = 10

print("Generating CPI of return pulses...")
for i in range(0,m):
    X,M,f,amp = tk.chirped_waveform_single(np.sqrt(rdr.P_t),sample_rate,chirp.freq,chirp.BW,chirp.tau,chirp.PRF)
    X_tx = tk.awgn(X,rdr.P_n,rdr.P_t)
    Y = tk.return_pulse(tgt.range,tgt.rate,tgt.RCS,rdr.P_t,rdr.G,rdr.L_s,rdr.P_n,X_tx,f,chirp.tau,sample_rate)
    if i == 0:
        coherent_sum = Y.copy()
    else:
        coherent_sum = np.add(Y,coherent_sum)
    del X,f,amp,X_tx,Y
    print(i+1," of ",m," pulses coherently summed.")
print("Complete with CPI integration.")

print("Beginning filter bank generation...")

#f_d_optial = -2*tgt.rate / chirp.wavelength
f_d_lower_unambiguous = -chirp.PRF
f_d_upper_unambiguous = chirp.PRF
k = 50 #number of doppler bins

i = 0
for f_d in np.linspace(f_d_lower_unambiguous,f_d_upper_unambiguous,k):
    i += 1
    M_doppler_shifted = M.copy()
    for n in range(0,len(M_doppler_shifted)):
        t = n/sample_rate
        M_doppler_shifted[n] = np.multiply(np.exp(-1j*2*np.pi*f_d*t),M_doppler_shifted[n])   
    print(i," of ",k," matched filters generated at f_d = ",f_d," Hz.")
    fast_time = np.correlate(coherent_sum,M_doppler_shifted)
    pwr_ft = 10*np.log10(np.multiply(fast_time,np.conjugate(fast_time)))
    if i == 1:
        l = len(fast_time)
        range_doppler = np.transpose(pwr_ft)
    else:
        range_doppler = np.column_stack(range_doppler,np.transpose(pwr_ft))
    print(i," of ",k," fast-time doppler bins calculated.")
    del M_doppler_shifted,fast_time,pwr_ft
del i

mesh_l,mesh_k = np.meshgrid(np.array(range(0,l)),np.array(range(0,k)))

#cs = plt.contourf(ft_fb_dB)
cs = plt.contourf(np.abs(range_doppler))
plt.colorbar(cs)
plt.title('Range-Doppler Plot')
plt.show()







# # generate an array for signal
# X,M,f,amp = tk.chirped_waveform_single(np.sqrt(rdr.P_t),sample_rate,chirp.freq,chirp.BW,chirp.tau,chirp.PRF)
# del amp
# # apply noise to the output
# X_tx = tk.awgn(X,rdr.P_n,rdr.P_t)
# pwr_X = 10*np.log10(np.multiply(X_tx,np.conjugate(X_tx)))
# spec_X = 10*np.log10(np.abs(np.fft.fft(X_tx)))
# # get the return pulse
# Y = tk.return_pulse(tgt.range,tgt.rate,tgt.RCS,rdr.P_t,rdr.G,rdr.L_s,rdr.P_n,X_tx,f,chirp.tau,sample_rate)
# pwr_Y = 10*np.log10(np.multiply(Y,np.conjugate(Y)))
# spec_Y = 10*np.log10(np.abs(np.fft.fft(Y)))
# # range process the pulse
# fast_time = np.correlate(Y,M)
# pwr_ft = 10*np.log10(np.multiply(fast_time,np.conjugate(fast_time)))


# # plot the clean and noisy signals on top of each other
# fig, axs = plt.subplots(3)
# s_X = np.array(range(0,len(pwr_X)))
# s_X = s_X/sample_rate
# s_Y = np.array(range(0,len(pwr_Y)))
# s_Y = s_Y/sample_rate
# fig.suptitle('Signal Time and Freq Domain Analysis')
# axs[0].set_title("Signal Power")
# axs[0].plot(
#     s_X,pwr_X,'b',
#     s_Y,pwr_Y,'r')
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Signal Power [dBW]")
# axs[0].xaxis.set_minor_locator(AutoMinorLocator())
# axs[0].yaxis.set_minor_locator(AutoMinorLocator())
# axs[0].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
# del s_X,s_Y
# # plot the clean and noisy signals on top of each other
# s_X = np.array(range(0,len(spec_X)))
# s_X = s_X*sample_rate/len(spec_X)
# s_Y = np.array(range(0,len(spec_Y)))
# s_Y = s_Y*sample_rate/len(spec_Y)
# axs[1].set_title("FFT Output")
# axs[1].plot(
#     s_X,spec_X,'b',
#     s_Y,spec_Y,'r')
# axs[1].set_xlabel("Sampled Frequency")
# #axs[1].set_ylim([-20,80])
# axs[1].set_ylabel("Power [dB]")
# axs[1].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
# axs[1].xaxis.set_minor_locator(AutoMinorLocator())
# axs[1].yaxis.set_minor_locator(AutoMinorLocator())
# del s_X,s_Y
# # plot the fast time response
# s_ft = np.array(range(0,len(pwr_ft)))
# s_ft = scipy.constants.c*s_ft/(2*sample_rate)
# axs[2].set_title("Fast Time Response")
# axs[2].plot(
#     s_ft,pwr_ft,'g')
# axs[2].set_xlabel("Range [m]")
# axs[2].set_ylabel("Power [dB]")
# axs[2].grid(visible=True, which='both',axis='both',color='0.8', linestyle='-', linewidth=1)
# axs[2].xaxis.set_minor_locator(AutoMinorLocator())
# axs[2].yaxis.set_minor_locator(AutoMinorLocator())
# plt.show()







#################################### UNCLASSIFIED ####################################
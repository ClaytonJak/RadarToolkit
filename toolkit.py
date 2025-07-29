from pickle import FALSE, TRUE
import numpy as np
import scipy.constants

# Constants
c = scipy.constants.c; #m/s, speed of light
k = scipy.constants.k; #J/K, Boltzmann constant

# Note: do not remove awgn() from the same file as signal_power()
def awgn(X,noise_power,sig_power):
    # description - this adds white gaussian noise to a numpy array signal
    # input X is the input numpy array 1xN
    # input is the power of WGN desired in dB
    # input sig_power is the pulse signal power in dB
    # output Y is numpy array 1xN with additave white gaussian noise
    Y = X.copy()
    sig_power_lin = np.power(10,sig_power/10)
    noise_cov = np.power(10,noise_power/10)
    sigma = np.sqrt(noise_cov)
    for n in range(0,len(Y)):
        WGN = complex(np.random.normal(0,sigma)) + complex(1j * np.random.normal(0, sigma))
        Y[n] += WGN
    return Y

def signal_power(X):
    # description - this calculates the power of a numpy array signal
    # input X is the 1xN numpy array signal
    # output pwr is a float average output power of the signal
    L = len(X)
    pwr = np.sum(np.power(X,2))/L
    return pwr

def CW_waveform(amp,t_len,sample_rate,freq):
    # description - this generates a continuous wave (CW) waveform in a 1xN numpy array
    # input amp is a float/int for desired CW amplitude
    # input t_len is a float/int for the desired time length (in seconds) of the CW waveform
    # input sample_rate is the sample rate (in samples/sec) of the sampled CW waveform
    # input freq is the frequency of the CW waveform (in Hz)
    # output X is the 1xN numpy array containing the CW waveform
    n_samples = int(t_len*sample_rate)
    X = np.zeros(n_samples, dtype = 'complex_')
    w_t = 2*np.pi*freq
    for n in range(0,len(X)):
        t = n/sample_rate
        X[n] = amp*np.exp(1j*w_t*t)
    return X,freq,amp

def pulsed_waveform_single(amp,sample_rate,freq,pulse_width,PRF):
    # description - this generates a single pulsed waveform in a 1xN numpy array
    # input amp is a float/int for the desired CW amplitude
    # input sample_rate is the sample rate (in samples/sec) of the sampled pulsed waveform
    # input freq is the center frequency of the pulsed waveform (in Hz)
    # input pulse_width is the pulse width of the pulsed waveform (in seconds)
    # input PRF is the pulse repitition frequency (PRF) of the pulsed waveform
    # output X is a 1xN numpy array containing one single PRI of the pulsed waveform
    # output M is a 1x(pulse width) array containing only the pulsed portion of the pulsed waveform (for matched filter purposes)
    PRI = 1/PRF
    n_samples = int(PRI*sample_rate)
    X = np.zeros(n_samples, dtype = 'complex_')
    w_t = 2*np.pi*freq
    marker = 0
    for n in range(0,len(X)):
        t = n/sample_rate
        if t < pulse_width:
            ampl = amp
            marker = n
        else:
            ampl = 0
        X[n] = ampl*np.exp(1j*w_t*t)
    M = X[:marker]
    return X,M,freq,amp

def pulsed_waveform_multi(amp,t_len,sample_rate,freq,pulse_width,PRF):
    # description - this generates a specified time sample of a pulsed waveform in a 1xN numpy array
    # input amp is a float/int for the desired amplitude
    # input t_len is a float/int for the desired time lenth of the output waveform (in seconds)
    # input sample_rate is the sample rate (in samples/sec) of the sampled pulsed waveform
    # input freq is the center frequency of the pulsed waveform (in Hz)
    # input pulse_width is the pulse width of the pulsed waveform (in seconds)
    # input PRF is the pulse repitition frequency (PRF) of the pulsed waveform
    # output X is a 1xN numpy array containing one single PRI of the pulsed waveform
    n_samples = int(t_len*sample_rate)
    X = np.zeros(n_samples, dtype = 'complex_')
    w_t = 2*np.pi*freq
    PRI = 1/PRF
    in_pulse_flag = FALSE
    PRI_multiple = 0
    for n in range(0,len(X)):
        t = n/sample_rate
        if in_pulse_flag == FALSE:
            if t > (PRI*PRI_multiple):
                in_pulse_flag = TRUE
                PRI_multiple += 1
                finish_pulse = t + pulse_width
        if in_pulse_flag == TRUE:
            if t < finish_pulse:
                ampl = amp
            else:
                in_pulse_flag = FALSE
                ampl = 0
        else:
            ampl = 0
        X[n] = ampl*np.exp(1j*w_t*t)
    return X,freq,amp

def chirped_waveform_single(amp,sample_rate,freq,BW,pulse_width,PRF):
    # description - this generates a single chirped waveform in a 1xN numpy array
    # input amp is a float/int for the desired amplitude
    # input sample_rate is the sample rate (in samples/sec) of the sampled chirped waveform
    # input freq is the center frequency of the chirped waveform (in Hz)
    # input BW is the 3dB bandwidth of the chirped waveform (in Hz)
    # input pulse_width is the pulse width of the chirped waveform (in seconds)
    # input PRF is the pulse repitition frequency (PRF) of the chirped waveform
    # output X is a 1xN numpy array containing one single PRI of the chirped waveform
    # output M is a 1x(pulse width) array containing only the pulsed portion of the chirped waveform (for matched filter purposes)
    PRI = 1/PRF
    n_samples = int(PRI*sample_rate)
    X = np.zeros(n_samples, dtype = 'complex_')
    w_t = 2*np.pi*freq
    w_t_lower = 2*np.pi*(freq-(BW/2))
    marker = 0
    for n in range(0,len(X)):
        t = n/sample_rate
        if t < pulse_width:
            ampl = amp
            marker = n
        else:
            ampl = 0
            W_t = w_t
        X[n] = ampl*np.exp(1j*((w_t_lower*t)+(np.pi*(BW/pulse_width)*(np.power(t,2)))))
    M = X[:marker]
    return X,M,freq,amp

def return_pulse(truth_range, truth_range_rate,truth_RCS,radar_P_t,radar_G,radar_L_s,radar_P_n,X,freq,SNR,sample_rate):
    # description - takes a TX waveform, reduces its power by free space path loss, RCS return, 
    #               and additional desired losses, then applies WGN at a desired SNR to the 
    #               return signal. Doppler shifts the return according to the target rate.
    #               Zero pads the beginning of the waveform to express the range.
    #               Returns a RX waveform with the signal buried in the noise.
    # input truth_range is the truth range to the target in meters
    # input truth_range_rate is the truth range rate to the target in m/s. Closing neg, opening pos.
    # input truth_RCS is the truth Swerling 0 RCS (in dBsm)
    # input radar_P_t is the transmit power of the radar (in dBW)
    # input radar_G is the radar antenna gain (in dBi)
    # input radar_P_n is the radar noise power (in dBW)
    # input radar_L_s is a place to account for different losses (in dB) not in free-space
    #               path loss (e.g. TX/RX chain, atmospherics, etc). Positive values are losses.
    # input X is the transmitted waveform
    # input SNR is the desired SNR (in dB) of the RX waveform (after losses) to WGN
    # input sample_rate is the sample rate of the TX waveform (will also be applied to RX waveform)
    # output Y is the RX waveform (will likely look like a signal buried in noise)
    Y = X.copy()
    # append zeros for target range
    t_delay = 2*truth_range/c
    n_delay = t_delay*sample_rate
    delay_array = np.zeros(int(n_delay))
    Y = np.append(delay_array,Y)
    # calculate the received signal power based on radar range equation
    wavelength = c/freq
    num = radar_P_t + (2*radar_G) + 20*np.log10(wavelength) + truth_RCS
    den = 30*np.log10(4*np.pi) + 40*np.log10(truth_range) + radar_L_s
    P_r_dB = num - den
    # apply doppler shift
    #f_d = 2*truth_range_rate / wavelength
    f_d = 10e6
    for n in range(0,len(Y)): 
        t = n/sample_rate
        Y[n] = np.multiply(np.exp(-1j*2*np.pi*f_d*t),Y[n])
    # scale the waveform by the power
    power_ratio_dB = radar_P_t - P_r_dB
    power_ratio_lin = np.power(10,power_ratio_dB/10)
    amplitude_ratio_lin = np.sqrt(power_ratio_lin)
    Y = Y/amplitude_ratio_lin
    # add noise to each return value
    Y = awgn(Y,radar_P_n,P_r_dB)
    # add clutter model here
    return Y
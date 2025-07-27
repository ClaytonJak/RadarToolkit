from pickle import FALSE, TRUE
import numpy as np

# Note: do not remove awgn() from the same file as signal_power()
def awgn(X,SNR_dB):
    Y = X.copy()
    SNR_lin = np.power(10,SNR_dB/10)
    noise_cov = signal_power(X)/SNR_lin
    sigma = np.sqrt(noise_cov)
    for n in range(0,len(Y)):
        mu = Y[n]
        Y[n] = np.random.normal(mu,sigma)
    return Y

def signal_power(X):
    L = len(X)
    pwr = np.sum(np.power(X,2))/L
    return pwr

def CW_waveform(amp,t_len,sample_rate,freq):
    n_samples = int(t_len*sample_rate)
    X = np.zeros(n_samples, dtype = 'complex_')
    w_t = 2*np.pi*freq
    for n in range(0,len(X)):
        t = n/sample_rate
        X[n] = amp*np.exp(1j*w_t*t)
    return X

def pulsed_waveform(amp,t_len,sample_rate,freq,pulse_width,PRF):
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
    return X
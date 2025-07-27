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
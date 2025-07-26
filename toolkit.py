import numpy as np

# Note: do not remove awgn() from the same file as signal_power()
def awgn(X,SNR_dB):
    Y = X
    SNR_lin = np.power(10,SNR_dB/10)
    noise_cov = signal_power(X)/SNR_lin

    return noise_pwr

def signal_power(X):
    L = len(X)
    pwr = np.sum(np.power(X,2))/L
    return pwr
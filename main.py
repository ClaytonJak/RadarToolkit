import numpy as np
import matplotlib.pyplot as plt
import toolkit

# generate an array for signal
size_test_sig = 100
test_sig = np.zeros(size_test_sig)
test_sig[int(size_test_sig/2)] = 1

# add WGN to the signal with a SNR of 6dB
noisy_sig = toolkit.awgn(test_sig,6)

# plot the clean and noisy signals on top of each other



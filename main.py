import numpy as np
import matplotlib.pyplot as plt
import toolkit

# seed random numbers numpy
np.random.seed(0)

# initialize matplotlib parameters
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# generate an array for signal
size_test_sig = 100
test_sig = np.zeros(size_test_sig)
test_sig[int(size_test_sig/2)] = 1

# add WGN to the signal with a SNR of 6dB
noisy_sig = toolkit.awgn(test_sig,6)

# plot the clean and noisy signals on top of each other
s = range(0,len(noisy_sig))
fig, axs = plt.subplots()
axs.set_title("Signal")
axs.plot(s, test_sig, s, noisy_sig)
axs.set_xlabel("Sample")
axs.set_ylabel("Amplitude")
plt.show()
del s


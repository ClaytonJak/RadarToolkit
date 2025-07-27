import numpy as np
import matplotlib.pyplot as plt
import toolkit

# seed random numbers numpy
np.random.seed(0)

# initialize matplotlib parameters
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# initialize sample rate
sample_rate = 5e9 #samples/sec

# generate an array for signal
X = toolkit.CW_waveform(1,5e-6,sample_rate,1e9)
spec = np.fft.fft(X)

# plot the clean and noisy signals on top of each other
#s = range(0,len(noisy_sig))
fig, axs = plt.subplots()
axs.set_title("Signal")
axs.plot(spec,'b')
#axs.plot(s, noisy_sig,'r')
axs.set_xlabel("Sample")
axs.set_ylabel("Amplitude")
plt.show()
del s


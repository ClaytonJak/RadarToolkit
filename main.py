import numpy as np
import matplotlib.pyplot as plt
import toolkit

# seed random numbers numpy
np.random.seed(0)

# initialize matplotlib parameters
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# initialize sample rate
sample_rate = 2e9 #samples/sec

# generate an array for signal
X = toolkit.pulsed_waveform(1,5e-3,sample_rate,1e9,100e-6,1e3)
pwr_X = np.abs(X)
#spec = np.fft.fft(X)

# plot the clean and noisy signals on top of each other
s = range(0,len(pwr_X))
fig, axs = plt.subplots()
axs.set_title("Signal")
#axs.plot(pwr_X,'b')
axs.plot(spec,'r')
axs.set_xlabel("Sample")
axs.set_ylabel("Amplitude")
plt.show()
del s


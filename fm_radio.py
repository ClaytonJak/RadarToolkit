#################################### UNCLASSIFIED ####################################
# 
# fm_radio.py
# Clayton Jaksha
# ALL RIGHTS RESERVED
# 
# all work completed on personally owned devices with personally owned or licensed software

from scipy.io import wavfile
import toolkit as tk
import scipy.constants
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


# Load the audio file
#sample_rate, data = wavfile.read('woman.wav')
sample_rate, data = wavfile.read('Yamaha-TG100-Ocarina-C5.wav')
#data_index = np.arange(0,10000,1)
#sample_rate = 25000  # Example sample rate
#data = np.sin(2*np.pi*5000*data_index/sample_rate)  # Example: 5 kHz sine wave
data = np.append(data, np.zeros(1000))  # Append zeros to the end of the data
data = np.append(np.zeros(1000), data)  # Append zeros to the beginning of the data
data_t = np.linspace(0,len(data)/sample_rate,len(data))

print(f"Sample Rate: {sample_rate} Hz")
print(f"Data Type: {data.dtype}")  # e.g., int16, float32


# Create the frequency spectrum
data_fft = np.fft.fft(data)
freq_bins = np.fft.fftfreq(len(data), d=1/sample_rate)
pos_freq_bins = freq_bins[freq_bins >= 0]
data_fft_dB = 20*np.log10((data_fft[freq_bins >= 0]))

figure, ax = plt.subplots(2,1)
ax[0].plot(data_t,data)
ax[0].set_title('Time Domain Signal')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Amplitude')

# Plot the frequency domain signal
ax[1].plot(pos_freq_bins, data_fft_dB)
ax[1].set_title('Frequency Domain Signal')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Power [dB]')


plt.show()
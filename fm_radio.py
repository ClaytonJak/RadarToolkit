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
from scipy import signal
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

# Create the frequency spectrum
data_fft = np.fft.fft(data)
freq_bins = np.fft.fftfreq(len(data), d=1/sample_rate)
pos_freq_bins = freq_bins[freq_bins >= 0]
data_fft_dB = 20*np.log10((data_fft[freq_bins >= 0]))

fig2, ax1 = plt.subplots(6,1,figsize=(11, 7))
ax1[0].plot(data_t,data)
ax1[0].set_title('Time Domain Signal')
ax1[0].set_xlabel('Time [s]')
ax1[0].set_ylabel('Amplitude')

# Plot the frequency domain signal
ax1[1].plot(pos_freq_bins, data_fft_dB)
ax1[1].set_title('Frequency Domain Signal')
ax1[1].set_xlabel('Frequency [Hz]')
ax1[1].set_ylabel('Power [dB]')

new_sample_rate = 1.5e6
data_resampled = signal.resample(data, int(len(data) * new_sample_rate / sample_rate))
data_resampled_phase = np.cumsum(data_resampled)/new_sample_rate
beta = 1
t = np.linspace(0, len(data_resampled)/new_sample_rate, len(data_resampled))
dt = 1/new_sample_rate
f_c = 500e3  # Carrier frequency in Hz
phase = (2*np.pi*t*f_c)+(2*np.pi*beta*data_resampled_phase)
fm_sig = tk.awgn(np.cos(phase), -20)  # Add noise to the FM signal

# Plot the resampled signal
ax1[2].plot(t, data_resampled,'g')
ax1[2].set_title('Resampled Signal')
ax1[2].set_xlabel('Time [s]')
ax1[2].set_ylabel('Amplitude')

# Plot the frequency domain of the resampled signal
data_resampled_fft = np.fft.fft(data_resampled)
freq_bins_resampled = np.fft.fftfreq(len(data_resampled), d=1/new_sample_rate)
pos_freq_bins_resampled = freq_bins_resampled[freq_bins_resampled >= 0]
data_resampled_fft_dB = 20*np.log10((data_resampled_fft[freq_bins_resampled >= 0]))

ax1[3].plot(pos_freq_bins_resampled, data_resampled_fft_dB,'g')
ax1[3].set_title('Frequency Domain Signal (Resampled)')
ax1[3].set_xlabel('Frequency [Hz]')
ax1[3].set_ylabel('Power [dB]')

# Plot the FM signal
ax1[4].plot(t, fm_sig,'m')
ax1[4].set_title('FM Signal')
ax1[4].set_xlabel('Time [s]')
ax1[4].set_ylabel('Amplitude')

# Plot the frequency domain of the resampled signal
fm_sig_fft = np.fft.fft(fm_sig)
freq_bins_fm_sig = np.fft.fftfreq(len(fm_sig), d=1/new_sample_rate)
pos_freq_bins_fm_sig = freq_bins_fm_sig[freq_bins_fm_sig >= 0]
fm_sig_fft_dB = 20*np.log10((fm_sig_fft[freq_bins_fm_sig >= 0]))

ax1[5].plot(pos_freq_bins_fm_sig, fm_sig_fft_dB,'m')
ax1[5].set_title('Frequency Domain Signal (FM Signal)')
ax1[5].set_xlabel('Frequency [Hz]')
ax1[5].set_ylabel('Power [dB]')

fig2,ax2 = plt.subplots(2,1,figsize=(11, 7))

analytical_signal = signal.hilbert(fm_sig)
instantaneous_phase = np.unwrap(np.angle(analytical_signal))
carrier_phase = 2*np.pi*t*f_c
demod_phase = instantaneous_phase - carrier_phase
message = np.diff(demod_phase)/(2*np.pi*beta*dt)
message = np.append(message,0)  # Append a zero to maintain the same length

# Plot the FM signal
ax2[0].plot(t, analytical_signal,'m')
ax2[0].set_title('FM Signal')
ax2[0].set_xlabel('Time [s]')
ax2[0].set_ylabel('Amplitude')

ax2[1].plot(t, message,'g')
ax2[1].set_title('FM Signal')
ax2[1].set_xlabel('Time [s]')
ax2[1].set_ylabel('Amplitude')

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram
from scipy.signal.windows import hann
import toolkit as tk

# Waveform parameters
pulse_duration = 30e-6  # 30 microseconds
bandwidth = 100e6  # 100 MHz
sample_rate = 4 * bandwidth  # 400 MHz sampling rate
t = np.arange(0, pulse_duration, 1/sample_rate)

# Generate linear FM chirp (LFM) pulse
# Start frequency at -50 MHz and sweep to +50 MHz (100 MHz span)
f0 = -bandwidth / 2
f1 = bandwidth / 2
waveform = chirp(t, f0, pulse_duration, f1, method='linear')

# Apply Hann window to reduce spectral leakage
#waveform *= hann(len(waveform))

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Time domain waveform
axes[0].plot(t * 1e6, waveform)
axes[0].set_xlabel('Time (µs)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Transmit Waveform - Time Domain')
axes[0].grid(True)

# Plot 2: Zoomed time domain (first 5 microseconds)
zoom_idx = np.where(t <= 5e-6)[0]
axes[1].plot(t[zoom_idx] * 1e6, waveform[zoom_idx])
axes[1].set_xlabel('Time (µs)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Transmit Waveform - Time Domain (Zoomed to 5 µs)')
axes[1].grid(True)

# Plot 3: STFT spectrogram
f, t_stft, Sxx = spectrogram(waveform, fs=sample_rate, window='hann',nperseg=512, noverlap=256)
Sxx_db = 10 * np.log10(Sxx + 1e-10)
im = axes[2].pcolormesh(t_stft * 1e6, f / 1e6, Sxx_db, shading='gouraud', cmap='viridis')
axes[2].set_ylabel('Frequency (MHz)')
axes[2].set_xlabel('Time (µs)')
axes[2].set_title('STFT Spectrogram')
axes[2].set_ylim([00, 60])
cbar = plt.colorbar(im, ax=axes[2])
cbar.set_label('Power (dB)')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt



# --- Example Usage ---
num_ranges = 200
num_doppler = 128
gamma_db = -15.0 # Typical wooded area
grazing_angle = 5.0 # Degrees

clutter_map = tk.generate_constant_gamma_clutter(num_ranges, num_doppler, gamma_db, grazing_angle)

# Plotting the Range-Doppler Clutter map
plt.figure(figsize=(8, 6))
plt.imshow(clutter_map, aspect='auto', cmap='plasma', extent=[-60, 60, num_ranges, 0])
plt.colorbar(label='Clutter Power (dBsm/m²)')
plt.title('Constant Gamma Clutter Model (Range-Doppler)')
plt.ylabel('Range Bins')
plt.xlabel('Doppler (Hz)')
plt.show()

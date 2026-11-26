import matplotlib.pyplot as plt
import numpy as np

def plot_samples(waveform, sampling_rate):
    time = np.arange(len(waveform)) / sampling_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform, label='Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Addition of Sine Generators with Random Noise')
    plt.legend()
    plt.show()
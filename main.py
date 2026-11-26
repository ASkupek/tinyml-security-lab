import numpy as np
from examples.plot_samples import plot_samples
from src.tinyml_security_lab.data import generate_sample
from src.tinyml_security_lab.splitting import split_dataset

# Parameters for the sine generators
seed = 42
rng = np.random.default_rng(seed)
freq = 1.0  # Frequency of the first sine wave generator
amplitude = 1.0  # Amplitude of the first sine wave generator

# Parameters for the waveform and noise
duration = 1.0  # Duration of the waveform in seconds
sampling_rate = 128  # Number of samples per second
noise_amplitude = 0.2  # Amplitude of the random noise

waveforms = []
labels = []

for i in range(0,1000):
    anomaly = rng.choice([True, False])
    # Generate the waveform
    waveform, label = generate_sample(freq=freq, amplitude=amplitude, duration=duration, sampling_rate=sampling_rate, rng=rng, anomaly=anomaly, noise_amplitude=noise_amplitude)
    waveforms.append(waveform)
    labels.append(label)


counts = np.bincount(labels, minlength=2)
print(f"Normal with noise only: {counts[0]}, with anomaly: {counts[1]}")


X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(waveforms=waveforms, labels=labels, seed=seed)
plot_samples(waveforms[0], sampling_rate)
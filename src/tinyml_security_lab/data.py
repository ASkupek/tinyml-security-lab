import numpy as np

def generate_sample(freq, amplitude, duration, sampling_rate,noise_amplitude, rng, anomaly=False):
    time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    sin_wave = amplitude * np.sin(2 * np.pi * freq * time)

    noise = rng.normal(0, noise_amplitude, size=sin_wave.shape)
    noise_wave = sin_wave + noise

    if anomaly:
        anomaly_index = rng.integers(0, len(noise_wave))
        noise_wave[anomaly_index] += 2.0

    label = 1 if anomaly else 0
    return noise_wave.astype(np.float32), label
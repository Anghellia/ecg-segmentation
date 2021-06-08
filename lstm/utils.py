import numpy as np
from scipy.signal import cspline1d, cspline1d_eval


def get_smoothed_signal(raw_signal):
    out0 = np.convolve(raw_signal, np.ones(5, dtype=float), 'valid') / 5
    r = np.arange(1, 5 - 1, 2)
    start = np.cumsum(raw_signal[:5 - 1])[::2] / r
    stop = (np.cumsum(raw_signal[:-5:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))

def update_input(signals):
    N, SIZE = signals.shape
    new_signals = np.zeros((N, 2, SIZE))
    for i, signal in enumerate(signals):
        normalized = (signal - signal.min()) / (signal.max() - signal.min())
        derivative = np.gradient(get_smoothed_signal(signal))
        new_signals[i] = np.stack((normalized, derivative))
    return new_signals

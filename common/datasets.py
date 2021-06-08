import torch
import numpy as np
from wfdb import processing


class TrainDataset:
    def __init__(self, signals, masks, add_noise):
        self.signals = np.expand_dims(signals, axis=1) if signals.ndim == 2 else signals
        self.size = signals.shape[-1]
        self.masks = masks
        self.add_noise = add_noise

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, i):
        start, end = self.get_random_window()
        signal = self.signals[i, :, start:end]
        mask = self.masks[i, start:end]
        if self.add_noise and np.random.binomial(1, p=0.4):
            signal += np.random.normal(0, 0.005, signal.shape[1])
        return torch.FloatTensor(signal), torch.LongTensor(mask)

    def get_random_window(self):
        window = 800
        shift = np.random.randint(0, self.size - window)
        return (shift, shift + window)


class TestDataset:
    def __init__(self, signals, masks, dataset, do_resample, add_noise):
        self.signals = np.expand_dims(signals, axis=1) if signals.ndim == 2 else signals
        self.masks = masks
        self.dataset = dataset
        self.do_resample = do_resample
        self.add_noise = add_noise

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, i):
        signal = self.signals[i, :, :]
        mask = self.masks[i, :]
        if self.add_noise:
            signal += np.random.normal(0, 0.005, signal.shape[1])
        if self.do_resample:
            fs, fs_target = (250, 500) if self.dataset == 'qtdb' else (500, 250)
            signal = processing.resample_sig(signal[0], fs, fs_target)[0]
            size = len(signal)
            signal = np.expand_dims(signal, axis=0)
            mask = self.resample(mask, size, fs_target / fs)
        return torch.FloatTensor(signal), torch.LongTensor(mask)
    
    def resample(self, mask, size, ratio): # works only for 1-channel data
        p_indexes = (ratio * np.where(mask == 1)[0]).astype(np.int64)
        qrs_indexes = (ratio * np.where(mask == 2)[0]).astype(np.int64)
        t_indexes = (ratio * np.where(mask == 3)[0]).astype(np.int64)
        mask = np.zeros(size)
        mask[p_indexes] = 1
        mask[qrs_indexes] = 2
        mask[t_indexes] = 3
        return mask

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


## Train visualization ##

class LossAccVisualizer:
    def __init__(self):
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
    
    def add_train_phase(self, loss, accuracy):
        self.train_loss_history.append(loss)
        self.train_accuracy_history.append(accuracy)

    def add_val_phase(self, loss, accuracy):
        self.val_loss_history.append(loss)
        self.val_accuracy_history.append(accuracy)

    def draw_results(self):
        plt.figure(figsize=(20,10))
        plt.plot(np.arange(1, len(self.train_loss_history) + 1), self.train_loss_history, label='train loss')
        plt.plot(np.arange(1, len(self.val_loss_history) + 1), self.val_loss_history, label='val loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.legend()
        plt.show()

        plt.figure(figsize=(20,10))
        plt.plot(np.arange(1, len(self.train_accuracy_history) + 1), self.train_accuracy_history, label='train acc')
        plt.plot(np.arange(1, len(self.val_accuracy_history) + 1), self.val_accuracy_history, label='val acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.title('Accuracy during training')
        plt.legend()
        plt.show()


class LossVisualizer:
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []
    
    def add_train_phase(self, loss):
        self.train_loss_history.append(loss)

    def add_val_phase(self, loss):
        self.val_loss_history.append(loss)

    def draw_results(self):
        plt.figure(figsize=(20, 10))
        plt.plot(np.arange(1, len(self.train_loss_history) + 1), self.train_loss_history, label='train loss')
        plt.plot(np.arange(1, len(self.val_loss_history) + 1), self.val_loss_history, label='val loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Loss during training')
        plt.legend()
        plt.show()


## ECG visualization ##

def plot_dilated_ecg(signals, masks, index, title='Dilated ECG-signal', dataset='ludb'):
    wave_type_to_color = {0:'k', 1:'y', 2:'r', 3:'g'}
    signal, mask = signals[index], masks[index]
    signal_size = signals.shape[1]
    sample_rate = 500 if dataset == 'ludb' else 250
    signal_duration = signal_size / sample_rate
    x = np.linspace(0, signal_duration, signal_size)
    # set up colors
    colors = [wave_type_to_color[label] for label in mask]
    # convert time series to line segments
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], signal[:-1], x[1:], signal[1:])]
    colored_lines = LineCollection(lines, colors=colors, linewidths=(2,))
    # plot data
    plt.figure(figsize=(18,5))
    ax = plt.axes()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.show()

def plot_ecg_with_domain_knowledge(signal):
    plt.figure(figsize=(18,5))
    plt.plot(signal[0])
    plt.plot(signal[1], color='y')
    plt.plot(signal[2], color='r')
    plt.plot(signal[3], color='g')
    plt.title('ECG-signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.show()

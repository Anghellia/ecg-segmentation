import numpy as np
import pandas as pd
from .utils import mask_to_delineation


class ECGMetrics:
    def __init__(self, waveform, sample_rate):
        self.onset_TP, self.onset_FP, self.onset_FN = 0, 0, 0
        self.offset_TP, self.offset_FP, self.offset_FN = 0, 0, 0
        self.waveform = waveform
        # for LUDB neighborhood is 75 points, for QTDB - 38 points
        self.tolerance = 150 / (1000 / sample_rate)

    def compute_metrics(self, pred, true):
        # Find 1 type errors
        for (pred_onset, pred_offset) in pred[self.waveform]:
            is_onset_okay = False
            is_offset_okay = False
            for (truth_onset, truth_offset) in true[self.waveform]:
                if not is_onset_okay and abs(pred_onset - truth_onset) <= self.tolerance:
                    self.onset_TP += 1
                    is_onset_okay = True
                if not is_offset_okay and abs(pred_offset - truth_offset) <= self.tolerance:
                    self.offset_TP += 1
                    is_offset_okay = True
                if is_onset_okay and is_offset_okay:
                    break
            if not is_onset_okay:
                self.onset_FP += 1
            if not is_offset_okay:
                self.offset_FP += 1

        # Find 2 type errors
        for (truth_onset, truth_offset) in true[self.waveform]:
            is_onset_okay = False
            is_offset_okay = False
            for (pred_onset, pred_offset) in pred[self.waveform]:
                if not is_onset_okay and abs(truth_onset - pred_onset) <= self.tolerance:
                    #self.onset_TP += 1
                    is_onset_okay = True
                if not is_offset_okay and abs(truth_offset - pred_offset) <= self.tolerance:
                    #self.offset_TP += 1
                    is_offset_okay = True
                if is_onset_okay and is_offset_okay:
                    break
            if not is_onset_okay:
                self.onset_FN += 1
            if not is_offset_okay:
                self.offset_FN += 1

    def get_result(self):
        onset_recall = self.onset_TP / (self.onset_TP + self.onset_FN)
        onset_precision = self.onset_TP / (self.onset_TP + self.onset_FP)
        onset_F1 = 2 * onset_recall * onset_precision / (onset_recall + onset_precision)
        onset_metrics = [onset_recall, onset_precision, onset_F1]

        offset_recall = self.offset_TP / (self.offset_TP + self.offset_FN)
        offset_precision = self.offset_TP / (self.offset_TP + self.offset_FP)
        offset_F1 = 2 * offset_recall * offset_precision / (offset_recall + offset_precision)
        offset_metrics = [offset_recall, offset_precision, offset_F1]

        return onset_metrics, offset_metrics


def compute_metrics(preds, truths, sample_rate):

    p_metrics = ECGMetrics('p', sample_rate)
    qrs_metrics = ECGMetrics('qrs', sample_rate)
    t_metrics = ECGMetrics('t', sample_rate)

    for idx, true_mask in enumerate(truths):
        pred_mask = preds[idx]
        mask_true_del = mask_to_delineation(true_mask)
        mask_pred_del = mask_to_delineation(pred_mask)
        for wave in (p_metrics, qrs_metrics, t_metrics):
            wave.compute_metrics(mask_pred_del, mask_true_del)

    results = pd.DataFrame(columns=['Metric', 'P_onset', 'P_offset', 'QRS_onset', 'QRS_offset', 'T_onset', 'T_offset'])
    p_results = p_metrics.get_result()
    qrs_results = qrs_metrics.get_result()
    t_results = t_metrics.get_result()
    for metric, idx in zip(['Recall', 'Precision', 'F1_score'], [0, 1, 2]):
        results = results.append({'Metric': metric, 'P_onset': p_results[0][idx], 'P_offset': p_results[1][idx],
                                  'QRS_onset': qrs_results[0][idx], 'QRS_offset': qrs_results[1][idx],
                                  'T_onset': t_results[0][idx], 'T_offset': t_results[1][idx]}, ignore_index=True)
    return results

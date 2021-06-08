import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from lstm.model import ConvLSTM
from lstm.utils import update_input
from common.datasets import TestDataset
from common.metrics import compute_metrics
from common.utils import normalize, remove_small, merge_small


def get_test_loader(args):
    signals_test = np.load(args.dataset + "_signals_test.npy")
    masks_test = np.load(args.dataset + "_masks_test.npy")

    signals_test = update_input(signals_test) 

    test_dataset = TestDataset(signals_test, masks_test, args.dataset, args.resample, args.add_noise)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, type=str, choices=('ludb', 'qtdb'),
                        help='Required. Choose dataset to eval performance: ludb or qtdb.')
    parser.add_argument('-add_noise', default=False, action='store_true',
                        help='Optional. Add normal noise to the input.')
    parser.add_argument('-resample', default=False, action='store_true',
                        help='Optional. Resample signal to train sample rate.')
    return parser

def eval(test_loader, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConvLSTM(2)
    model.load_state_dict(torch.load('lstm_model'), strict=False)
    model = model.to(device)
    model.eval()
    
    all_masks_pred = []
    all_masks_truth = []

    for signal, mask_truth in test_loader:
        signal = signal.to(device)
        mask_truth = mask_truth.numpy()[0]
        all_masks_truth.append(mask_truth)

        with torch.set_grad_enabled(False):
            mask = model(signal)[0]
        mask = mask.max(axis=0)[1].data.cpu().numpy()

        all_masks_pred.append(mask)

    model.train()
    fs = 500 if dataset == 'ludb' else 250
    eval_res = compute_metrics(all_masks_pred, all_masks_truth, fs)
    return eval_res

def main():
    args = build_argparser().parse_args()
    test_loader = get_test_loader(args)
    df = eval(test_loader, args.dataset)
    print(df)

if __name__ == "__main__":
    main()

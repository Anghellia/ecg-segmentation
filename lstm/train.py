import argparse
import torch
import numpy as np
from IPython.display import clear_output
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from lstm.model import ConvLSTM
from lstm.utils import update_input
from common.visualize import LossAccVisualizer
from common.datasets import TrainDataset
from common.utils import normalize

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-add_noise', default=False, action='store_true',
                        help='Optional. Add normal noise to the input.')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Optional. Input batch size for training. Default: 1')
    parser.add_argument('-epochs', type=int, default=50,
                        help='Optional. Number of epochs to train model. Default: 50')
    parser.add_argument('-learning_rate', type=float, default=0.1,
                        help='Optional. Learning rate of optimizer. Default: 0.1')
    return parser

def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    signals, masks = np.load("qtdb_signals_train.npy"), np.load("qtdb_masks_train.npy")
    signals_train, signals_val, masks_train, masks_val = train_test_split(signals, masks)

    signals_train, signals_val = update_input(signals_train), update_input(signals_val)  

    train_loader = DataLoader(TrainDataset(signals_train, masks_train, args.add_noise),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TrainDataset(signals_val, masks_val, args.add_noise),
                            batch_size=args.batch_size, shuffle=True)
    train(train_loader, val_loader, args)

def train(train_loader, val_loader, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConvLSTM(2)
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    visualizer = LossAccVisualizer()
    best_model = None
    best_acc = None

    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #clear_output(True)
            print(f"Starting epoch: {epoch+1}/{args.epochs} | phase: {phase} ")
            if phase == 'train':
                loader = train_loader
                model.train()  # Set model to training mode
            else:
                loader = val_loader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(loader)
            epoch_acc = running_acc / len(loader)

            if phase == 'train':
                visualizer.add_train_phase(epoch_loss, epoch_acc)
            else:
                visualizer.add_val_phase(epoch_loss, epoch_acc)
                
                if best_acc is None or best_acc < epoch_acc:
                    best_acc = epoch_acc
                    best_model = model
                
            print(f"Loss: {'%.4f' % epoch_loss} | Acc: {'%.4f' % epoch_acc}\n")
    visualizer.draw_results()
        
    print(f"Best model performed {'%.4f' % (best_acc * 100)}% accuracy on validation data.")
    torch.save(best_model.state_dict(), 'lstm_model')

if __name__ == "__main__":
    main()

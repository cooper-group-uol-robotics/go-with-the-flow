import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch

# ------------------- label conversion tools ------------------ ##


def labels2cat(label_encoder, lst):
    return label_encoder.transform(lst)


def labels2onehot(onehotencoder, label_encoder, lst):
    return onehotencoder.transform(label_encoder.transform(lst).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


# ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        x_out = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder,
                                            f'frame{i:01d}.jpg')).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            x_out.append(image.squeeze_(0))
        x_out = torch.stack(x_out, dim=0)

        return x_out

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        # (input) spatial images
        x_out = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)
        # (labels) LongTensor are for int64 instead of FloatTensor
        y_out = torch.LongTensor([self.labels[index]])

        # print(x_out.shape)
        return x_out, y_out

# ---------------------- end of Dataloaders ---------------------- ##

# ------------------------ 3D CNN module ---------------------- ##


def conv3d_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] -
                          (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] -
                          (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] -
                          (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

# --------------------- end of 3D CNN module ---------------- ##


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving \
                      model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

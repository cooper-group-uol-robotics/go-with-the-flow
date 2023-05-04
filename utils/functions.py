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

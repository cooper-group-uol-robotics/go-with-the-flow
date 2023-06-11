import os
import re
from typing import Tuple, List

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..utils.functions import Dataset3DCNN, labels2cat
from ..transforms.transform import create_loading_transform


def create_reg_datasets(batch_size: int, data_path: str,
                        selected_frames: List[int],
                        img_x: int, img_y: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the training, testing and validation data loaders for training
    regression models. Note that the each training video is stored inside a
    folder with the folder named with its corresponding viscosity value.

    Args:
        batch_size: batch size used for training the model
        selected_frames: list of the selected frames indices of the training video
        int_x: x dimension to transform the input image for training
        int_y: y dimension to transform the input image for training
    """

    all_x_list = []
    all_y_list = []

    for filename in os.listdir(data_path):
        number = re.search(r'\d+', filename).group(0)
        all_x_list.append(number)
        all_y_list.append(int(number))

    # percentage of training set to be used for validation
    validation_size = 0.2

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_x_list, all_y_list,
                                                                      test_size=0.2, random_state=0)

    transform = create_loading_transform(img_x, img_y)
    train_set = Dataset3DCNN(data_path, train_list, train_label, selected_frames,
                             transform=transform)
    test_set = Dataset3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)

    # split into training and validation batches
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    return train_loader, test_loader, valid_loader


def create_cls_datasets(batch_size: int, data_path: str,
                        class_labels: List[str], selected_frames: List[int],
                        img_x: int, img_y: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the training, testing and validation data loaders for training classification models
    Note that the each training video is stored inside a folder with the folder named with its
    corresponding viscosity value.

    Args:
        batch_size: batch size used for training the model
        class_labels: list of strings containing the data labels
        selected_frames: list of the selected frames indices of the training video
        int_x: x dimension to transform the input image for training
        int_y: y dimension to transform the input image for training
    """

    # convert labels -> category
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    # # convert category -> 1-hot
    # viscosity_cats = label_encoder.transform(class_labels).reshape(-1, 1)
    # one_hot_enc = OneHotEncoder()
    # one_hot_enc.fit(viscosity_cats)

    all_x_list = []
    all_y_list = []
    viscosities = []

    for filename in os.listdir(data_path):
        loc1 = filename.find('v_')
        loc2 = filename.find('_g')
        all_x_list.append(filename)
        viscosities.append(filename[(loc1 + 2): loc2])

    # create labels data
    all_y_list = labels2cat(label_encoder, viscosities)

    # percentage of training set to be used for validation
    validation_size = 0.2

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_x_list, all_y_list,
                                                                      test_size=0.2, random_state=0)

    transform = create_loading_transform(img_x, img_y)
    train_set = Dataset3DCNN(data_path, train_list, train_label, selected_frames,
                             transform=transform)
    test_set = Dataset3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)

    # split into training and validation batches
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    return train_loader, test_loader, valid_loader

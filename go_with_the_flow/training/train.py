import os
from typing import Tuple, List

import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from pytorchtools import EarlyStopping

from ..models.models import CNN3DVisco


def train(
    model: CNN3DVisco,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    batch_size: int,
    learning_rate: float,
    patience: int,
    epochs: int,
    save_model_path: str,
    device: torch.device
) -> Tuple[CNN3DVisco, List[float], List[float]]:
    """ Train the model using the given input parameters using Adam optimiser and cross
    entropy loss function

    Args:
        model: CNN3DVisco model to be trained
        train_loader: training data loader
        valid_loader: validation data loader
        batch_size: training batch size
        learning_rate: training learning rate
        patience: training patience
        epochs: max training epochs
        save_model_path: path to save training outputs
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    # train
    # set model as training mode
    model.train()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    n_count = 0   # counting total trained sample in one epoch

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):

        for _batch_idx, (x_p, y_p) in enumerate(train_loader):

            epoch_subtract = 0
            # distribute data to device
            x_p, y_p = x_p.to(device), y_p.to(device).view(-1, )
            n_count += x_p.size(0)
            optimizer.zero_grad()
            output = model(x_p)
            loss = criterion(output, y_p)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # validate the model
        model.eval()
        for (x_p, y_p) in valid_loader:
            # distribute data to device
            x_p, y_p = x_p.to(device), y_p.to(device).view(-1, )
            output = model(x_p)
            loss = criterion(output, y_p)
            valid_losses.append(loss.item())                 # sum up batch loss

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (
            f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
            f'train_loss: {train_loss:.5f} ' +
            f'valid_loss: {valid_loss:.5f}'
        )

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            epoch_subtract += 1
            print("Early stopping")
            break
    print('epoch', epoch)
    print('epoch subtract', patience)
    summation = epoch - patience
    print(summation)

    # save spatial_encoder
    torch.save(
        model.state_dict(),
        os.path.join(
            save_model_path,
            'lr_' + str(learning_rate) + 'batch_' + str(batch_size) + f'3dcnn_epoch{summation}.pth'
        )
    )
    # save optimizer
    torch.save(
        optimizer.state_dict(),
        os.path.join(
            save_model_path,
            str(batch_size) + f'3dcnn_optimizer_epoch{(summation)}.pth'
        )
    )
    print(f"Epoch {summation} model saved!")

    return model, avg_train_losses, avg_valid_losses


def test_model(
    model: CNN3DVisco,
    test_loader: DataLoader,
    batch_size: int,
    num_classes: int
) -> Tuple[float, List[float], List[float]]:
    """ Test the given model using the test dataset and return overall accuracy in addition to
    the true and predicted values

    Args:
        model: CNN3DVisco model to be trained
        test_loader: test data loader
        batch_size: training batch size
        num_classes: number of classification labels
    """

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = [0. for i in range(num_classes)]
    class_total = [0. for i in range(num_classes)]
    criterion = nn.CrossEntropyLoss()

    # prep model for evaluation
    model.eval()

    # lists for confusion matrix
    true_values = []
    predicted_values = []

    for _batch_indx, (data, target) in enumerate(test_loader):
        target1 = target.data[:, 0]
        print('target1.data', len(target1.data))
        print('batch size', batch_size)
        if len(target1.data) != batch_size:
            break

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target1)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        print('pred', pred)

        # compare predictions to the true label
        correct = np.squeeze(pred.eq(target1.data.view_as(pred)))

        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target1.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            label = target1.data[i]
            label_integer = label.item()
            true_values.append(label_integer)

            pred_integer = pred.view(pred.numel()).numpy()
            predicted_values.append(pred_integer[0])
            # predicted_values.append(pred)

    print('true values', true_values)
    print('pred values', predicted_values)

    # calculate and print the avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}\n')

    for i in range(num_classes):
        if class_total[i] > 0:
            print(
                'Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i),
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i])
                )
            )

    print(
        '\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total)
        )
    )

    overall_accuracy = 100 * np.sum(class_correct) / np.sum(class_total)
    print('overall accuracy = ', overall_accuracy)
    return overall_accuracy, true_values, predicted_values

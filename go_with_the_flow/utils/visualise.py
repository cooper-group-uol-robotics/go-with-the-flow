import os
from typing import List, Any

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn


def visualise_loss_early_stop(train_loss: List[float], valid_loss: List[float],
                              save_path: str, save_filename: str,
                              show_fig: bool = False) -> None :
    """ Generate training and validation loss plot along with early stop

    Args:
    train_loss: list containing training loss across training epochs
    valid_loss: list containing validation loss across training epochs
    save_path: path where the figure will be saved
    save_filename: saved figure filename
    show_fig: if True show figure
    """
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    min_pos = valid_loss.index(min(valid_loss))+1
    plt.axvline(min_pos, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 10)  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if show_fig:
        plt.show()
    file_name = os.path.join(save_path, f"{save_filename}.png")
    fig.savefig(file_name, bbox_inches='tight')


def visualise_confusion_matrix(pred_values: List[Any], true_values: List[Any],
                               title: str, x_label: str, y_label: str, tick_labels: List[str],
                               save_path: str, save_filename: str,
                               show_fig: bool = False) -> None:
    """ Generate confusion matrix for the given data

    Args:
    pred_values: list containing model data label predictions
    true_values: list containing data real labels
    title: figure title
    x_label: x axis label
    y_label: y axis label
    tick_labels: class label to be used as figure ticks
    save_path: path where the figure will be saved
    save_filename: saved figure filename
    show_fig: if True show figure
    """
    conf_mat = confusion_matrix(pred_values, true_values)
    ax = plt.subplot()

    sn.heatmap(conf_mat, annot=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(tick_labels)
    ax.yaxis.set_ticklabels(tick_labels)
    if show_fig:
        plt.show()
    file_name = os.path.join(save_path, f"{save_filename}.png")
    plt.savefig(file_name)

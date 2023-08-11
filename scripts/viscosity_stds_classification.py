import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import cuda, device, manual_seed
from go_with_the_flow.models.models import CNN3DVisco
from go_with_the_flow.data.loader import create_cls_datasets
from go_with_the_flow.training.train import train, test_model
from go_with_the_flow.utils.visualise import visualise_loss_early_stop, visualise_confusion_matrix

if __name__ == "__main__":
    # use CPU for running
    use_cuda = cuda.is_available()
    dev = device("cuda" if use_cuda else "cpu")

    # setting the random seeds
    manual_seed(0)
    cuda.manual_seed(0)
    np.random.seed(0)

    epochs = 1500
    patience = 30
    batch_size = 32
    learning_rate = 0.1
    data_path = "163_VS_classify"
    target_save_path = "Conv3D_ckpt_VC_final"
    class_labels = ["v_low", "low", "medium", "high", "v_high"]
    begin_frame, end_frame, skip_frame = 30, 400, 40
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    img_x, img_y = 256, 342

    # construct model
    model = CNN3DVisco(t_dim=29, img_x=img_x, img_y=img_y, num_classes=len(class_labels))

    # load datasets
    train_loader, test_loader, valid_loader = create_cls_datasets(
        batch_size, data_path, class_labels, selected_frames, img_x, img_y
    )

    # train model
    model, train_loss, valid_loss = train(
        model, train_loader, valid_loader, batch_size, learning_rate, patience, epochs, target_save_path, dev
    )

    # visualise training loss
    fig_save_path = "./figs"
    loss_filename = "loss_plot"
    visualise_loss_early_stop(train_loss, valid_loss, fig_save_path, loss_filename)

    # test trained model
    _, true_values, predicted_values = test_model(model, test_loader, batch_size, len(class_labels))

    # plot confusion matrix
    plt_title = "Confusion Matrix"
    x_label = "Predicted Viscosity Category"
    y_label = "Actual Viscosity Category"
    conf_mat_filename = "conf_mat"
    visualise_confusion_matrix(
        predicted_values, true_values, plt_title, x_label, y_label, class_labels, fig_save_path, conf_mat_filename
    )

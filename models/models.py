import torch.nn as nn
import torch.nn.functional as func

from ..utils.functions import conv3d_output_size


class CNN3DVisco(nn.Module):
    def __init__(self, t_dim: int = 30, img_x: int = 256 , img_y: int = 342, drop_p: int = 0,
                 fc_hidden1: int = 256, fc_hidden2: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3d_output_size((self.t_dim, self.img_x, self.img_y), self.pd1,
                                                 self.k1, self.s1)
        self.conv2_outshape = conv3d_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1,
                               stride=self.s1, padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2,
                               stride=self.s2, padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        # fully connected hidden layer
        self.fc1 = nn.Linear(self.ch2*self.conv2_outshape[0]*self.conv2_outshape[1] *
                             self.conv2_outshape[2], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # fully connected layer, output = multi-classes
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)

    def forward(self, x_3d):
        # Conv 1
        x_out = self.conv1(x_3d)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        x_out = self.drop(x_out)
        # Conv 2
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)
        x_out = self.drop(x_out)
        # FC 1 and 2
        x_out = x_out.view(x_out.size(0), -1)
        x_out = func.relu(self.fc1(x_out))
        x_out = func.relu(self.fc2(x_out))
        x_out = func.dropout(x_out, p=self.drop_p, training=self.training)
        x_out = self.fc3(x_out)
        return x_out

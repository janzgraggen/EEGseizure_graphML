# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-05-13 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Implement a CNN-*-

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# import files
import constants
from models.base_model import BaseModel


class CNN(BaseModel):
    """Implementation of a Convolutional Neural Network (CNN) for
    time series classification.
    """

    def __init__(
        self,
        input_shape,
        device=constants.DEVICE,
    ):
        super().__init__(device=device)
        self.input_shape = input_shape
        self.layers = None
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0], out_channels=32, kernel_size=2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.dropout_conv = nn.Dropout(0.2)

        # Calculate the size of the flattened layer after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1])
            x = self.forward_conv(dummy_input)
            flatten_size = x.shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, 64)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward_conv(self, x):
        """Implement the forward pass for the convolutional layers.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).
        Returns:
            torch.Tensor: Output tensor after convolutional layers.
        """
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

    def forward(self, x):
        """Forward pass of the CNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.forward_conv(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_fc(x)
        return x

    @staticmethod
    def from_config(model_cfg):
        return CNN(**model_cfg)

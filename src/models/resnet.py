# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-05-20 by Caspar -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-


# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import mlflow
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader

# import files
import constants
from train import *
from models.classic_base import ClassicBase


class ResNet(ClassicBase):
    """
    Code taken from https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
    and adapted using Pytorch instead of Keras

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, input_shape: int, mid_channels: int = 64,
                 output_shape: int = 1) -> None:
        super().__init__()

        self.device = constants.DEVICE
        self.input_shape = input_shape
        self.output_shape = output_shape

        n_feature_maps = mid_channels

        self.block1 = ResidualBlock(self.input_shape, n_feature_maps, expand_channels=True)
        self.block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, expand_channels=True)
        self.block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, expand_channels=False)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # replaces GlobalAveragePooling1D
        self.fc = nn.Linear(n_feature_maps * 2, self.output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_avg_pool(out)
        out = out.squeeze(-1)  # remove last dim (B, C, 1) → (B, C)
        out = self.fc(out)
        return out
    

    @staticmethod
    def from_config(model_cfg):
        """Create a model from a configuration dictionary.

        Args:
            model_cfg (dict): Configuration dictionnary

        Returns:
            CnnBase: The model
        """

        return ResNet(**model_cfg)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        if expand_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        shortcut = self.shortcut(x)
        if out.size(2) != shortcut.size(2):
            shortcut = F.pad(shortcut, (0, out.size(2) - shortcut.size(2)))
        out += shortcut
        out = F.relu(out)
        return out

    

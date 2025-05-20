# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-05-20 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Implement a fcn model-*-

# Import libraries
import torch.nn as nn

# Import files
import constants
from models.base_model import BaseModel

class FCN(BaseModel):
    def __init__(self, input_channels, device=constants.DEVICE):
        super().__init__(device=device)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(128, 1)

        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.conv3,
            self.bn3,
            nn.ReLU(),
            self.global_avg_pool,
            nn.Flatten(),
            self.fc
        )
        self.to(device)

    @staticmethod
    def from_config(model_cfg):
        return FCN(**model_cfg)
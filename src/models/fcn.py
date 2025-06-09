# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Implement a fcn model-*-

# Import libraries
import torch.nn as nn

# Import files
import constants
from models.base_model import BaseModel

class FCN(BaseModel):
    def __init__(self, num_classes=2, input_channels=19):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.softmax = nn.Softmax(dim=(1,1))

        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

        self.to(constants.DEVICE)

    @staticmethod
    def from_config(model_cfg):
        return FCN(**model_cfg)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)

        x = self.classifier(x)
        x = self.global_avg_pool(x)  
        x = x.squeeze(-1)         
        return x

        
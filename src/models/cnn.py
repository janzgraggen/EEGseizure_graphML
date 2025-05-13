# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-05-13 -*-
# -*- Last revision: 2025-05-13 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Implement a CNN-*-

# Import libraries
import torch

# import files
import constants
from models.base_model import BaseModel

class CNN(BaseModel):
    def __init__(
            self,
            input_shape,
            device=constants.DEVICE,
    ):
        super().__init__(device=device)
        self.input_shape = input_shape
        self.layers = None
        self._build_model()
    
    def _build_model(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_shape[0], self.input_shape[1])
            feature_extractor = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=self.input_shape[0], out_channels=6, kernel_size=7),
                torch.nn.Sigmoid(),
                torch.nn.AvgPool1d(kernel_size=3),
                torch.nn.Conv1d(in_channels=6, out_channels=12, kernel_size=7),
                torch.nn.Sigmoid(),
                torch.nn.AvgPool1d(kernel_size=3),
                torch.nn.Flatten(),
            )
            dummy_output = feature_extractor(dummy_input)
            n_features = dummy_output.shape[1]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_shape[0], out_channels=6, kernel_size=7),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=6, out_channels=12, kernel_size=7),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool1d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout1d(p=0.2),
            torch.nn.Linear(in_features=n_features, out_features=1),
        )
        self.layers.to(self.device)

    @staticmethod
    def from_config(model_cfg):
        return CNN(**model_cfg)
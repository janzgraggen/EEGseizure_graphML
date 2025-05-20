# -*- coding: utf-8 -*-
# -*- authors : janzgraggen -*-
# -*- date : 2025-05-02 -*-
# -*- Last revision: 2025-05-06 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch_geometric.nn as nngc
import torch.nn as nn
import torch

# Import parent class and constants
from models.graph_base import GraphBase
import constants


"""
TO BE TESTED...
"""

class GAT(GraphBase):
    def __init__(self, in_channels: int, hidden_channels: int, heads=4):
        super().__init__()
        self.device = constants.DEVICE
        self.conv1 =  nngc.GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 =  nngc.GATConv(hidden_channels * heads, hidden_channels)
        self.lin =    nn.Linear(hidden_channels, 1)

        self.to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x =  nngc.global_mean_pool(x, batch)
        return self.lin(x)
    @staticmethod
    def from_config(model_cfg):
        return GAT(**model_cfg)

class GCN(GraphBase):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.device = constants.DEVICE
        self.conv1 =  nngc.GCNConv(in_channels, hidden_channels)

        self.conv2 =  nngc.GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

        self.to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x =  nngc.global_mean_pool(x, batch)
        return self.lin(x)
    
    @staticmethod
    def from_config(model_cfg):
        return GCN(**model_cfg)
    

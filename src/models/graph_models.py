# -*- coding: utf-8 -*-
# -*- authors : janzgraggen -*-
# -*- date : 2025-05-02 -*-
# -*- Last revision: 2025-05-26 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch_geometric.nn as nngc
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

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

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.dropout2 = nn.Dropout(0.2)

        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.5)

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        # Xavier for linear layer
        init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

        # Xavier for GCNConv layers
        for conv in [self.conv1, self.conv2]:
            init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                init.zeros_(conv.lin.bias)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout2(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.lin(x)

    @staticmethod
    def from_config(model_cfg):
        return GCN(**model_cfg)
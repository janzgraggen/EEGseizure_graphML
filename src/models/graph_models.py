# -*- coding: utf-8 -*-
# -*- authors : janzgraggen -*-
# -*- date : 2025-05-02 -*-
# -*- Last revision: 2025-06-01 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch_geometric.nn as nngc
import torch.nn as nn
import torch
import torch.nn.functional as F

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

        self.conv1 = nngc.GCNConv(in_channels, hidden_channels)
        self.bn1 = nngc.BatchNorm(hidden_channels)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nngc.GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nngc.BatchNorm(hidden_channels)
        self.dropout2 = nn.Dropout(0.2)

        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.5)

        self.leaky_relu_slope = 0.1
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

        for conv in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(conv.lin.weight)
            if conv.lin.bias is not None:
                nn.init.zeros_(conv.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.dropout1(x)

        x_res = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + x_res
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        x = self.dropout2(x)

        x = nngc.global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.lin(x)

    @staticmethod
    def from_config(model_cfg):
        return GCN(**model_cfg)


class LSTMGNN(GraphBase):
    def __init__(self, in_channels, hidden_channels_gcn, hidden_channels_lstm):
        super().__init__()
        self.device = constants.DEVICE
        self.gcn1 = nngc.GCNConv(in_channels, hidden_channels_gcn)
        self.gcn2 = nngc.GCNConv(hidden_channels_gcn, hidden_channels_gcn)
        self.lstm = nn.LSTM(
            input_size=hidden_channels_gcn,
            hidden_size=hidden_channels_lstm,
            num_layers=1,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_channels_lstm, 1)

        self.to(self.device)

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GCN layer with ReLU activation
        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)
        
        # Second GCN layer with ReLU activation
        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(x2)
        
        # Stack outputs to form a sequence of two time steps
        sequence = torch.stack([x1, x2], dim=0)  # Shape: [2, num_nodes, hidden_dim_gcn]
        
        # Process the sequence through LSTM
        lstm_out, _ = self.lstm(sequence)  # Shape: [2, num_nodes, hidden_dim_lstm]
        
        # Extract the last time step output for each node
        node_embeddings = lstm_out[-1]  # Shape: [num_nodes, hidden_dim_lstm]
        
        # Aggregate node embeddings to graph-level via mean pooling
        graph_embeddings = nngc.global_mean_pool(node_embeddings, batch)  # Shape: [batch_size, hidden_dim_lstm]
        
        # Final linear layer for binary classification logits
        logits = self.fc(graph_embeddings)  # Shape: [batch_size, 1]
        
        return logits

    @staticmethod
    def from_config(model_cfg):
        return LSTMGNN(**model_cfg)


class LSTMGAT(GraphBase):
    def __init__(self, in_channels, hidden_channels_gat, hidden_channels_lstm):
        super().__init__()
        self.device = constants.DEVICE
        # First GAT layer with 8 attention heads
        self.gat1 = nngc.GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels_gat,
            heads=8,  # Multi-head attention
            concat=False  # Average heads instead of concatenating
        )
        # Second GAT layer
        self.gat2 = nngc.GATConv(
            in_channels=hidden_channels_gat,
            out_channels=hidden_channels_gat,
            heads=8,
            concat=False
        )

        self.gat3 = nngc.GATConv(
            in_channels=hidden_channels_gat,
            out_channels=hidden_channels_gat,
            heads=8,
            concat=False
        )
        # LSTM to process sequence of GAT outputs
        self.lstm = nn.LSTM(
            input_size=hidden_channels_gat,
            hidden_size=hidden_channels_lstm,
            num_layers=1,
            batch_first=False
        )
        # Final classifier
        self.fc = nn.Linear(hidden_channels_lstm, 1)

        self.to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer with ELU activation
        x1 = self.gat1(x, edge_index)
        x1 = F.elu(x1)
        
        # Second GAT layer with ELU activation
        x2 = self.gat2(x1, edge_index)
        x2 = F.elu(x2)

        x3 = self.gat3(x2, edge_index)
        x3 = F.elu(x3)
        
        # Stack GAT outputs as sequence (2 timesteps)
        sequence = torch.stack([x1, x2, x3], dim=0)  # Shape: [2, num_nodes, hidden_dim_gat]
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(sequence)  # Shape: [2, num_nodes, hidden_dim_lstm]
        
        # Take last timestep output
        node_embeddings = lstm_out[-1]  # Shape: [num_nodes, hidden_dim_lstm]
        
        # Global mean pooling for graph-level embedding
        graph_embeddings = nngc.global_mean_pool(node_embeddings, batch)
        
        # Final binary classification
        logits = self.fc(graph_embeddings)
        
        return logits

    @staticmethod
    def from_config(model_cfg):
        return LSTMGAT(**model_cfg)

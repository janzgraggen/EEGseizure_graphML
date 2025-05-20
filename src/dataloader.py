# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-05-06 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to load the project-*-

# Import librairies
import pandas as pd
from PyQt5.sip import array
from holoviews.operation import threshold
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import networkx as nx
import copy  # for deep graph copy
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.convert import from_networkx
import torch
import numpy as np
from collections import Counter
import random
# Import modules
import constants
from transform_func import *


def parse_datasets(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parse the datasets from the configuration file.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data

    Returns:
        tuple: A tuple containing the loaders for train, validation, and test datasets.
    """

    num_datasets = len(cfg)
    if num_datasets == 0:
        raise ValueError("No datasets provided in the configuration file.")

    for i in tqdm(range(num_datasets), desc="Loading datasets", unit="dataset"):
        dataset = cfg[i]
        set_type = dataset.get("set", None)
        if set_type == "train":
            loader_train = load_data(cfg=dataset)
        elif set_type == "val":
            loader_val = load_data(cfg=dataset)
        elif set_type == "test":
            loader_test = load_data(cfg=dataset)
        else:
            raise ValueError(f"Unknown dataset type: {set_type}")
    return loader_train, loader_val, loader_test


def load_data(cfg: dict) -> DataLoader:
    """Load the data from the path and return a DataLoader.

    Args:
        cfg (dict): Configuration dictionary containing the informations about the data

    Returns:
        loader (DataLoader): A DataLoader object containing the data.
    """
    # Read clips
    path = cfg.get("path", None)
    if path is None:
        raise ValueError("No data path provided in the configuration file.")
    clips_path = os.path.join(path, "segments.parquet")
    clips = pd.read_parquet(clips_path)

    split = cfg.get("split", None)

    if split is not None:
        start = int(split[0] * len(clips))
        end = int(split[1] * len(clips))
        clips = clips.iloc[start:end]

    # Get transform
    tfm_name = cfg.get("tfm", None)
    tfm = get_transform(tfm_name=tfm_name)


    # Get additional parameters
    batch_size = cfg.get("batch_size", constants.BATCH_SIZE)
    shuffle = cfg.get("shuffle", True)
    set_name = cfg.get("set", None)
    get_id = True if set_name == "test" else False
    sampling = cfg.get("sampling", False)
    size = int(cfg.get("size", 1))

    # Create dataset
    dataset = EEGDataset(
        clips, signals_root=path, signal_transform=tfm, prefetch=True, return_id=get_id
    )

    sampler = None

    if sampling:
        n_classes = len(np.unique(dataset.get_label_array()))
        weights = make_weights_for_balanced_classes(dataset, n_classes)
        sampler = WeightedRandomSampler(weights, size * len(dataset), replacement=True)
        shuffle = False

    # Graph construction
    graph_cfg = cfg.get("graph", None)
    
    if graph_cfg is not None:
        dataset = graph_construction(dataset, graph_cfg, cfg)

        return torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    return loader

def make_weights_for_balanced_classes(samples:Dataset, nclasses:int) -> list:
    """Code taken from https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample
    This function is used to create weights for each class in the dataset.

    Args:
        samples (torch.utils.data.Dataset): Dataset containing the samples.
        nclasses (int): Number of classes in the dataset.

    Returns:
        weights (list): A list of weights for each sample in the dataset.
    """
    n_samples = len(samples)
    count_per_class = [0] * nclasses
    for _, sample_class in samples:
        count_per_class[sample_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_samples) / float(count_per_class[i])
    weights = [0] * n_samples
    for idx, (sample, sample_class) in enumerate(samples):
        weights[idx] = weight_per_class[sample_class]
    return weights

def graph_construction(dataset, graph_cfg, cfg):

    graph_type = graph_cfg.get("type", None)
    distance_path = graph_cfg.get("path", constants.DISTANCE_3D_FILE)
    distance_path = os.path.expandvars(distance_path)
    get_graph_summary = graph_cfg.get("get_graph_summary", None)
    
    data_set = cfg.get("set", None)

    if graph_type == 'distance':
        distance_df = pd.read_csv(distance_path)
        edge_threshold =  cfg.get("edge_threshold", None)

        adj_matrix = distance_df.pivot(index='from', columns='to', values='distance')
        adj_matrix = adj_matrix.reindex(distance_df['from'].unique(), axis=0).reindex(distance_df['from'].unique(),
                                                                                      axis=1)
        adj_matrix = np.where(adj_matrix < edge_threshold, adj_matrix, 0)

        distance_graphs = []
        for i in range(len(dataset)):

            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(dataset[i][0].T, dtype=np.float32)  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index) # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            distance_graphs.append(pyg_graph)

        return distance_graphs


    elif graph_type == 'correlation':

        edge_threshold =  cfg.get("edge_threshold", None)
        correlation_graphs = []

        for i in range(len(dataset)):
            adj_matrix = np.corrcoef(np.asarray(dataset[i][0].T, dtype=np.float32))  # shape: (n_channels, n_channels)
            adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)


            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(dataset[i][0].T, dtype=np.float32)  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index) # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            correlation_graphs.append(pyg_graph)

        return correlation_graphs

    else:
        raise ValueError("This graph construction method is not implemented.")

def get_transform(tfm_name: str) -> callable:
    """Get the transform function based on the name provided.

    Args:
        tfm_name (str): Name of the transform function to be used.

    Returns:
        function: The transform function corresponding to the name provided.
    """
    if tfm_name == "fft":
        return fft_filtering
    else:
        return None

def graph_signal_summary(signals):
    summary_array = np.stack([
        np.mean(signals, axis=1),
        np.median(signals, axis=1),
        np.max(signals, axis=1),
        np.min(signals, axis=1),
        np.std(signals, axis=1),
    ], axis=1)

    return summary_array
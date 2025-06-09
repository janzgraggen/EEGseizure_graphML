# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to load the project-*-

# Import librairies
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import torch_geometric
import torch
import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.signal import coherence
from sklearn.model_selection import StratifiedKFold

# Import modules
import constants
from transform_func import *
from utils import is_mostly_zero_record


def parse_datasets(
    datasets: list, config: dict, submission: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parse the datasets from the configuration file

    Args:
        datasets (list): List of dataset config dicts.
        config (dict): General config dictionary.
        submission (bool): If True, load only train and test datasets for submission.
                            If False, load train, val, and test datasets for training.

    Returns:
        tuple: DataLoaders for train, val, and test sets.
    """
    if len(datasets) == 0:
        raise ValueError("No datasets provided in the configuration file.")

    loader_train, loader_val, loader_test = None, None, None

    for i in tqdm(range(len(datasets)), desc="Loading datasets", unit="dataset"):
        dataset = datasets[i]
        set_type = dataset.get("set", None)
        if submission:
            if set_type == "train":
                loader_train = load_data(
                    dataset_cfg=dataset, config=config, submission=True
                )
            elif set_type == "test":
                loader_test = load_data(
                    dataset_cfg=dataset, config=config, submission=True
                )
            else:
                continue

        else:
            if set_type == "train":
                loader_train = load_data(dataset_cfg=dataset, config=config)
            elif set_type == "val":
                loader_val = load_data(dataset_cfg=dataset, config=config)
            elif set_type == "test":
                continue
            else:
                raise ValueError(f"Unknown dataset type: {set_type}")

    return loader_train, loader_val, loader_test


def load_data(dataset_cfg: dict, config: dict, submission: bool = False) -> list:
    """Load data and split according to fold for cross-validation.
    Args:
        dataset_cfg (dict): Configuration dictionary for the dataset.
        config (dict): General configuration dictionary.
        submission (bool): If True, load only train and test datasets for submission.
    Returns:
        list: List of DataLoaders for the dataset.
    """
    path = dataset_cfg.get("path", None)
    if path is None:
        raise ValueError("No data path provided in the configuration file.")

    # Read clips
    clips = pd.read_parquet(os.path.join(path, "segments.parquet"))

    # Apply transform
    tfm_name = config.get("tfm", None)
    tfm = get_transform(tfm_name=tfm_name)

    # Get dataset parameters
    batch_size = config.get("batch_size", constants.BATCH_SIZE)
    shuffle = dataset_cfg.get("shuffle", True)
    set_name = dataset_cfg.get("set", None)
    get_id = True if set_name == "test" else False
    sampling = dataset_cfg.get("sampling", False)
    size = int(dataset_cfg.get("size", 1))
    num_workers = config.get("num_workers", constants.NUM_WORKERS)
    n_splits = config.get("n_splits", 0)

    dataloaders = []

    # If submission is True, we only load the train set and a dummy validation set
    if submission:
        n_splits = 1

    # Get datasets for the specified splits
    for split in range(n_splits):
        print(f"len of clips: {len(clips)}")
        if set_name in ["train", "val"]:
            labels = clips["label"].values
            indices = np.arange(len(clips))

            if submission:
                train_idx = indices
                val_idx = [0]  # Dummy index for submission
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = list(skf.split(indices, labels))
                train_idx, val_idx = splits[split]

            if set_name == "train":
                clips_split = clips.iloc[train_idx].copy()
            elif set_name == "val":
                clips_split = clips.iloc[val_idx].copy()

            clips_split.sort_index(inplace=True)

        dataset = EEGDataset(
            clips_split,
            signals_root=path,
            signal_transform=tfm,
            prefetch=True,
            return_id=get_id,
        )

        # Remove zero-value samples
        valid_indices = [
            i
            for i in range(len(dataset))
            if not is_mostly_zero_record(dataset[i][0])
        ]

        dataset = torch.utils.data.Subset(dataset, valid_indices)
        sampler = None

        # Under-sampling or over-sampling
        if sampling:
            n_classes = len(np.unique(dataset.dataset.get_label_array()))
            weights = make_weights_for_balanced_classes(dataset, n_classes)
            sampler = WeightedRandomSampler(
                weights, size * len(dataset), replacement=True
            )
            shuffle = False

        # Get graph construction configuration
        graph_cfg = config.get("graph", None)
        
        if graph_cfg is not None:
            dataset, _ = graph_construction(dataset, graph_cfg, dataset_cfg)

            dataloader = torch_geometric.loader.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
            )
        dataloaders.append(dataloader)

    # If only one dataset is provided, return the single DataLoader
    if len(dataloaders) == 1:
        return dataloaders[0]
    else:
        return dataloaders


def make_weights_for_balanced_classes(samples: Dataset, nclasses: int) -> list:
    """Code taken from
    https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample
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
    weight_per_class = [0.0] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_samples) / float(count_per_class[i])
    weights = [0] * n_samples
    for idx, (sample, sample_class) in enumerate(samples):
        weights[idx] = weight_per_class[sample_class]
    return weights


def graph_construction(
    dataset: EEGDataset, graph_cfg: dict, cfg: dict
) -> tuple[list, np.ndarray]:
    """Construct graphs from the dataset based on the graph configuration.

    Args:
        dataset (EEGDataset): The dataset containing EEG signals.
        graph_cfg (dict): Configuration dictionary for the graph construction.
        cfg (dict): General configuration dictionary.

    Returns:
        tuple: A tuple containing the constructed graphs and the adjacency matrix.
    """

    # Fetch graph configuration parameters
    graph_type = graph_cfg.get("type", None)
    distance_path = graph_cfg.get("path", constants.DISTANCE_3D_FILE)
    distance_path = os.path.expandvars(distance_path)
    get_graph_summary = graph_cfg.get("get_graph_summary", None)

    data_set = cfg.get("set", None)

    # Construct distance graph
    if graph_type == "distance":
        distance_df = pd.read_csv(distance_path)
        edge_threshold = graph_cfg.get("edge_threshold", None)

        adj_matrix = distance_df.pivot(index="from", columns="to", values="distance")
        adj_matrix = (
            adj_matrix.reindex(distance_df["from"].unique(), axis=0)
            .reindex(distance_df["from"].unique(), axis=1)
            .to_numpy()
        )
        with np.errstate(divide="ignore"):
            adj_matrix = 1 / adj_matrix
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = (adj_matrix - np.min(adj_matrix)) / (
            np.max(adj_matrix) - np.min(adj_matrix)
        )

        adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)

        distance_graphs = []
        for i in range(len(dataset)):

            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(
                dataset[i][0].T, dtype=np.float32
            )  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(
                x=x_tensor, edge_index=edge_index
            )  # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            distance_graphs.append(pyg_graph)

        return distance_graphs, adj_matrix

    # Construct correlation
    elif graph_type == "correlation":
        edge_threshold = graph_cfg.get("edge_threshold", None)
        correlation_graphs = []
        correlation_adj_matrix = []

        for i in range(len(dataset)):
            data = np.asarray(dataset[i][0].T, dtype=np.float32)
            adj_matrix = safe_corrcoef(data)
            adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)

            if np.min(adj_matrix) == np.max(adj_matrix):
                adj_matrix = np.zeros_like(adj_matrix)
            else:
                adj_matrix = (adj_matrix - np.min(adj_matrix)) / (
                    np.max(adj_matrix) - np.min(adj_matrix)
                )

            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(
                dataset[i][0].T, dtype=np.float32
            )  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(
                x=x_tensor, edge_index=edge_index, edge_attr=edge_weight
            )  # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            correlation_graphs.append(pyg_graph)
            correlation_adj_matrix.append(adj_matrix)

        return correlation_graphs, correlation_adj_matrix

    # Construct coherence graph
    elif graph_type == "coherence":
        edge_threshold = graph_cfg.get("edge_threshold", None)
        coherence_energie_graphs = []
        coherence_adj_matrix = []

        for i in range(len(dataset)):

            f, coherence_matrix = signal.coherence(
                x=dataset[i][0][:, :, np.newaxis],
                y=dataset[i][0][:, np.newaxis, :],
                fs=250,
                axis=0,
            )
            coherence_matrix = np.nan_to_num(
                coherence_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            adj_matrix = coherence_matrix[np.logical_and(f >= 0.5, f <= 30), :, :].mean(
                axis=0
            )
            adj_matrix = np.where(adj_matrix >= edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)

            if np.min(adj_matrix) == np.max(adj_matrix):
                adj_matrix = np.zeros_like(adj_matrix)
            else:
                adj_matrix = (adj_matrix - np.min(adj_matrix)) / (
                    np.max(adj_matrix) - np.min(adj_matrix)
                )

            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(
                dataset[i][0].T, dtype=np.float32
            )  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals  # np.logical_and(freqs >= 0.5, freqs <= 30)

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(
                x=x_tensor, edge_index=edge_index, edge_attr=edge_weight
            )  # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            coherence_energie_graphs.append(pyg_graph)
            coherence_adj_matrix.append(adj_matrix)

        return coherence_energie_graphs, coherence_adj_matrix

    else:
        raise ValueError("This graph construction method is not implemented.")


def safe_corrcoef(X: np.ndarray, eps=1e-8):
    """Compute the correlation coefficient matrix of the input data,
        ensuring numerical stability.

    Args:
        X (np.ndarray): Input data of shape (n_channels, n_samples).
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: Correlation coefficient matrix of shape (n_channels, n_channels).
    """
    X = X - np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std < eps] = eps  # Avoid division by zero
    X_norm = X / std
    return np.dot(X_norm, X_norm.T) / X.shape[1]


def get_transform(tfm_name: str) -> callable:
    """Get the transform function based on the name provided.

    Args:
        tfm_name (str): Name of the transform function to be used.

    Returns:
        function: The transform function corresponding to the name provided.
    """
    if tfm_name == "fft":
        return fft_filtering
    elif tfm_name == "time":
        return time_filtering
    elif tfm_name == "clean":
        return clean_input
    elif tfm_name == "psd":
        return power_spectral_density
    else:
        return None


def graph_signal_summary(signals: np.ndarray) -> np.ndarray:
    """Compute a summary of the EEG signals for graph construction.
    Args:
        signals (np.ndarray): EEG signals of shape (n_channels, signal_len).
    Returns:
        np.ndarray: Summary array of shape (n_channels, 5) containing mean, median, max, min, and std.
    """
    summary_array = np.stack(
        [
            np.mean(signals, axis=1),
            np.median(signals, axis=1),
            np.max(signals, axis=1),
            np.min(signals, axis=1),
            np.std(signals, axis=1),
        ],
        axis=1,
    )

    return summary_array


def compute_band_energy(eeg_data: np.array, fs: int = 250) -> np.ndarray:
    """_summary_

    Args:
        eeg_data (np.ndarray): EEG data of shape (n_channels, signal_len).
        fs (int, optional): The sampling frequency of the EEG data. Defaults to 250.

    Returns:
        np.ndarray: Energy matrix of shape (n_channels, 4) containing energy in delta,
                    theta, alpha, and beta bands.
    """
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    n_channels = eeg_data.shape[0]
    energy_matrix = np.zeros((n_channels, 4))

    for ch in range(n_channels):
        # Welch returns PSD estimate
        freqs, psd = signal.welch(eeg_data[ch], fs=fs, nperseg=fs)

        for i, band in enumerate(freq_bands):
            idx_band = np.logical_and(
                freqs >= freq_bands[band][0], freqs <= freq_bands[band][1]
            )
            # Integrate PSD over band: power ≈ energy
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            energy_matrix[ch, i] = band_power

    return energy_matrix

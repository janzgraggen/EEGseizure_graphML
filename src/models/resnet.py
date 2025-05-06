# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-05-06 by Caspar -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-


# Import libraries
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import mlflow
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader

# import files
import constants
from train import *


class ResNetBaseline(nn.Module):
    """
    Code taken from https://github.com/okrasolar/pytorch-timeseries/blob/master/src/models/resnet_baseline.py
    A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

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

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))
    
    def fit(
        self,
        loader_tr,
        loader_val,
        num_epochs=constants.NUM_EPOCHS,
        learning_rate=constants.LEARNING_RATE,
        criterion_name=constants.CRITERION,
        optimizer_name=constants.OPTIMIZER,
    ):
        self.train_losses = []
        self.val_losses = []

        self.optimizer = get_optimizer(optimizer_name, self.parameters(), learning_rate)
        self.criterion = get_criterion(criterion_name)

        pbar = tqdm(total=num_epochs, desc="Training", position=0, leave=True)
        for e in range(num_epochs):
            # Training
            train_loss = self._epoch(loader_tr, train=True)
            self.train_losses.append(train_loss)
            accuracy, f1_score = self.predict(loader_tr)

            mlflow.log_metric("accuracy", accuracy, step=e + 1)
            mlflow.log_metric("f1_score 0", f1_score[0], step=e + 1)
            mlflow.log_metric("f1_score 1", f1_score[1], step=e + 1)
            mlflow.log_metric("train_loss", train_loss, step=e + 1)

            # Validation
            val_loss = self._epoch(loader_val, train=False)
            self.val_losses.append(val_loss)
            accuracy, f1_score = self.predict(loader_val)
            mlflow.log_metric("val_f1_score 0", f1_score[0], step=e + 1)
            mlflow.log_metric("val_f1_score 1", f1_score[1], step=e + 1)
            mlflow.log_metric("val_accuracy", accuracy, step=e + 1)
            mlflow.log_metric("val_loss", val_loss, step=e + 1)

            pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})
            pbar.update(1)

    def predict_batch(self, x):
        """Make a prediction on a batch of data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model predictions.
        """
        self.eval()
        with torch.no_grad():
            x = x.float().to(self.device)
            x = x.permute(0, 2, 1)
            logits = self(x)
            predictions = (logits > 0).int()
        return predictions

    def predict(self, loader):
        """Make a prediction on a dataset.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for the dataset.

        Returns:
            tuple: (accuracy, f1 score tensor)
        """
        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                batch_predictions = self.predict_batch(x_batch)

                # Flatten and move to CPU for consistency
                all_predictions.extend(batch_predictions.flatten().cpu().tolist())
                all_targets.extend(y_batch.flatten().cpu().tolist())

        # Convert lists to torch tensors
        all_predictions = torch.tensor(all_predictions).long()
        all_targets = torch.tensor(all_targets).long()

        # Compute accuracy
        correct = (all_predictions == all_targets).sum().item()
        accuracy = correct / len(all_targets)

        # Compute F1 score
        f1 = multiclass_f1_score(
            all_predictions,
            all_targets,
            num_classes=self.output_shape + 1,
            average=None,
        ).tolist()

        return accuracy, f1

    def _epoch(self, loader, train=True):

        if train:
            self.train()
        else:
            self.eval()
        running_loss = 0.0

        for x_batch, y_batch in loader:

            x_batch = x_batch.float().to(self.device)
            x_batch = x_batch.permute(0, 2, 1)
            y_batch = y_batch.float().unsqueeze(1).to(self.device)

            # Forward pass
            logits = self(x_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        return avg_loss

    def create_submission(
        self, loader: DataLoader, path: str = constants.SUBMISSION_FILE
    ):
        self.eval()
        # Lists to store sample IDs and predictions
        all_predictions = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:

                # Unpack the batch
                x_batch, x_ids = batch

                # permute the input tensor to match the expected shape
                x_batch = x_batch.permute(0, 2, 1)

                # Move to device
                x_batch = x_batch.float().to(self.device)

                # Perform the forward pass to get the model's output logits
                logits = self(x_batch)

                # Convert logits to predictions.
                predictions = (logits > 0).int().cpu().numpy()

                all_predictions.extend(predictions.flatten().tolist())
                all_ids.extend(list(x_ids))

        submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
        submission_df.to_csv(path, index=False)

        print(f"Submission file created at {path}")
        return submission_df

    @staticmethod
    def from_config(model_cfg):
        """Create a model from a configuration dictionary.

        Args:
            model_cfg (dict): Configuration dictionnary

        Returns:
            CnnBase: The model
        """

        return CnnBase(**model_cfg)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)

    

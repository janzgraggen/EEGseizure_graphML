# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-05-06 by Caspar -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch
from tqdm import tqdm
import pandas as pd
import mlflow
from torcheval.metrics.functional import binary_f1_score
from torch.utils.data import DataLoader

# import files
import constants
from train import *


class CnnBase(torch.nn.Module):

    def __init__(
        self,
        layers,
        input_shape=19,
        output_shape=2,
        kernel_size=5,
        device=constants.DEVICE,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_configs = layers
        self.kernel_size = kernel_size
        self.device = device

        layers = []
        in_channels = self.input_shape

        for layer_cfg in self.layer_configs:
            out_channels = layer_cfg["out_channels_multiplier"] * self.input_shape
            pooling_type = layer_cfg.get("pooling", "max")

            layers.append(
                torch.nn.Conv1d(
                    in_channels, out_channels, self.kernel_size, padding="same"
                )
            )
            layers.append(torch.nn.BatchNorm1d(out_channels))
            layers.append(torch.nn.ReLU())

            if pooling_type == "max":
                layers.append(torch.nn.MaxPool1d(2))
            elif pooling_type == "adaptiveavg":
                layers.append(torch.nn.AdaptiveAvgPool1d(1))
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")

            in_channels = out_channels

        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(in_channels, self.output_shape))

        self.layers = torch.nn.Sequential(*layers)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

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
            train_accuracy, train_f1_score = self.predict(loader_tr)

            mlflow.log_metric("train_f1_score ", train_f1_score, step=e + 1)
            mlflow.log_metric("train_accuracy", train_accuracy, step=e + 1)
            mlflow.log_metric("train_loss", train_loss, step=e + 1)

            # Validation
            val_loss = self._epoch(loader_val, train=False)
            self.val_losses.append(val_loss)
            val_accuracy, val_f1_score = self.predict(loader_val)
            
            mlflow.log_metric("val_f1_score ", val_f1_score, step=e + 1)
            mlflow.log_metric("val_accuracy", val_accuracy, step=e + 1)
            mlflow.log_metric("val_loss", val_loss, step=e + 1)

            pbar.set_postfix({"\ntrain_loss": train_loss, "val_loss": val_loss,
                "\ntrain_f1_score": train_f1_score, "val_f1_score": val_f1_score,
                "\ntrain_accuracy": train_accuracy, "val_accuracy": val_accuracy})
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
        f1 = binary_f1_score(
            all_predictions,
            all_targets
        )
        return accuracy, float(f1)

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

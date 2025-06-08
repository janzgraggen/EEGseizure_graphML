# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-06-08 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Implement the base model-*-

# Import libraries
import torch
from tqdm import tqdm
import pandas as pd
import mlflow
import tempfile
import os
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# import files
import constants
from train import *
from plots import plot_cm_matrix

class BaseModel(torch.nn.Module):
    def __init__(
        self,
        device=constants.DEVICE,
    ):
        super().__init__()
        self.device = device
        self.layers = None

    def forward(self, x):
        if self.layers is None:
            raise NotImplementedError("Layers not defined in the model.")
        return self.layers(x)

    def fit(
        self,
        loader_tr,
        loader_val,
        num_epochs=constants.NUM_EPOCHS,
        learning_rate=constants.LEARNING_RATE,
        criterion_name=constants.CRITERION,
        optimizer_name=constants.OPTIMIZER,
        use_scheduler=True,
    ):
        self.train_losses = []
        self.val_losses = []

        self.to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.parameters(), learning_rate)
        self.criterion = get_criterion(criterion_name)
        self.use_scheduler = use_scheduler

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',         
                factor=0.5,           
                patience=3,
            )

        pbar = tqdm(total=num_epochs, desc="Training", position=0, leave=True)
        for e in range(num_epochs):
            # Training
            train_loss = self._epoch(loader_tr, train=True)
            self.train_losses.append(train_loss)
            train_accuracy, train_f1_score, cm_train = self.predict(loader_tr)

            mlflow.log_metric("train_f1_score ", train_f1_score, step=e + 1)
            mlflow.log_metric("train_accuracy", train_accuracy, step=e + 1)
            mlflow.log_metric("train_loss", train_loss, step=e + 1)

            # Validation
            val_loss = self._epoch(loader_val, train=False)
            self.val_losses.append(val_loss)
            val_accuracy, val_f1_score, cm_val = self.predict(loader_val)
            
            if self.use_scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]
                mlflow.log_metric("learning_rate", current_lr, step=e + 1)

            with tempfile.TemporaryDirectory() as tmp_dir:
                os.makedirs(tmp_dir, exist_ok=True)
                plot_cm_matrix(
                    cm_train,
                    set="train",
                    file_pth=tmp_dir,
                    epoch=e + 1,
                )
                plot_cm_matrix(
                    cm_val,
                    set="val",
                    file_pth=tmp_dir,
                    epoch=e + 1,
                )
            
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
            predictions = (logits > 0).int().squeeze(1).cpu()
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
            average="macro",
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(
            all_targets.cpu(),
            all_predictions.cpu(),
        )

        return accuracy, float(f1), cm

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
            
            if train:
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

                # Move to device
                x_batch = x_batch.float().to(self.device)

                predictions = self.predict_batch(x_batch)

                all_predictions.extend(predictions.flatten().tolist())
                all_ids.extend(list(x_ids))

        submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
        submission_df.to_csv(path, index=False)

        print(f"Submission file created at {path}")
        return submission_df
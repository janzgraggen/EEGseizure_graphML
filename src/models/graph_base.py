# -*- coding: utf-8 -*-
# -*- authors : janzgraggen -*-
# -*- date : 2025-05-02 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to train models-*-

# Import libraries
from tqdm import tqdm
import pandas as pd
import mlflow
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader
import tempfile
import os
from sklearn.metrics import confusion_matrix


# import files
import constants
from train import *
from plots import plot_cm_matrix


class GraphBase(torch.nn.Module):
    """Base class providing training loop, evaluation, and submission. This class
    should be subclassed to implement specific graph neural network architectures.
    """

    def forward(self, x):
        """Implement in subclass"""
        raise NotImplementedError(...)

    def fit(
        self,
        loader_tr,
        loader_val,
        num_epochs=constants.NUM_EPOCHS,
        learning_rate=constants.LEARNING_RATE,
        criterion_name=constants.CRITERION,
        optimizer_name=constants.OPTIMIZER,
        submission=False,
        use_scheduler=True,
        fold=0
    ):
        """Train the model using the provided DataLoaders for training and validation.

        Args:
            loader_tr (DataLoader): DataLoader for training data.
            loader_val (DataLoader): DataLoader for validation data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            criterion_name (str): Name of the loss function to use.
            optimizer_name (str): Name of the optimizer to use.
            submission (bool): If True, skip validation and only train.
        """
        self.train_losses = []
        if not submission:
            self.val_losses = []

        self.optimizer = get_optimizer(optimizer_name, self.parameters(), learning_rate)
        self.criterion = get_criterion(criterion_name)
        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.5
                )
        
        self.to(self.device)

        pbar = tqdm(total=num_epochs, desc="Training", position=0, leave=True)
        for e in range(num_epochs):
            # Training
            train_loss = self._epoch(loader_tr, train=True)
            self.train_losses.append(train_loss)
            train_accuracy, train_f1_score, cm_train = self.predict(loader_tr)

            mlflow.log_metric(f"train_f1_score_{fold}", train_f1_score, step=e + 1)
            mlflow.log_metric(f"train_accuracy_{fold}", train_accuracy, step=e + 1)
            mlflow.log_metric(f"train_loss_{fold}", train_loss, step=e + 1)

            if not submission:
                # Validation
                val_loss = self._epoch(loader_val, train=False)
                self.val_losses.append(val_loss)
                val_accuracy, val_f1_score, cm_val = self.predict(loader_val)

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

                mlflow.log_metric(f"val_f1_score_{fold}", val_f1_score, step=e + 1)
                mlflow.log_metric(f"val_accuracy_{fold}", val_accuracy, step=e + 1)
                mlflow.log_metric(f"val_loss_{fold}", val_loss, step=e + 1)

                pbar.set_postfix(
                    {
                        "F1 (train)": train_f1_score,
                        f"\033[1m\033[31mF1 (VAL)\033[0m": val_f1_score,
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "F1 (train)": train_f1_score,
                    }
                )
            pbar.update(1)

            if self.use_scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                mlflow.log_metric("learning_rate", current_lr, step=e + 1)

    def predict_batch(self, batch):
        """Make predictions on a PyG batch.

        Args:
            batch (torch_geometric.data.Batch): A batch of graphs.
        Returns:
            torch.Tensor: Binary predictions (0 or 1).
        """
        self.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self(batch)
            predictions = (logits > 0).int()  # binary threshold
        return predictions.view(-1)

    def predict(self, loader):
        """Predict on an entire dataset of graphs.

        Args:
            loader (torch_geometric.data.DataLoader): Graph DataLoader.
        Returns:
            Tuple[float, List[float]]: (accuracy, per-class F1 scores)
        """
        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                predictions = self.predict_batch(batch)
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.view(-1).cpu().long())

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Compute accuracy
        accuracy = (all_predictions == all_targets).sum().item() / len(all_targets)

        all_predictions = all_predictions.long()
        all_targets = all_targets.long()

        # Compute F1 score
        f1 = multiclass_f1_score(
            all_predictions, all_targets, average="macro", num_classes=2
        )

        cm = confusion_matrix(
            all_targets.cpu(),
            all_predictions.cpu(),
        )
        return accuracy, float(f1), cm

    def _epoch(self, loader, train=True):
        """Run a single epoch of training or validation.

        Args:
            loader (DataLoader): DataLoader for the current epoch.
            train (bool): If True, run in training mode; otherwise, validation mode.

        Returns:
            float: Average loss for the epoch.
        """
        if train:
            self.train()
        else:
            self.eval()
        running_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            out = self(batch)  # expects batch to be PyG Batch object
            y = batch.y.view(-1, 1).float()

            loss = self.criterion(out, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def create_submission(
        self, loader: DataLoader, path: str = constants.SUBMISSION_FILE
    ):
        """Function to create a submission file from the predictions of the model.

        Args:
            loader (DataLoader): DataLoader for the test dataset.
            path (str, optional): Path to save the submission file.
                                    Defaults to constants.SUBMISSION_FILE.

        Returns:
            pd.DataFrame: DataFrame containing the submission data with
                        columns 'id' and 'label'.
        """
        self.eval()
        # Lists to store sample IDs and predictions
        all_predictions = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:
                # Unpack the batch
                x_ids = batch.id

                predictions = self.predict_batch(batch)

                all_predictions.extend(predictions.flatten().tolist())
                all_ids.extend(list(x_ids))

        submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
        submission_df.to_csv(path, index=False)

        print(f"Submission file created at {path}")
        return submission_df
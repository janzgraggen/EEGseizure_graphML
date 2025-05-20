# -*- coding: utf-8 -*-
# -*- authors : janzgraggen -*-
# -*- date : 2025-05-02 -*-
# -*- Last revision: 2025-05-06 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-

# Import libraries
#import torch
import torch_geometric
from tqdm import tqdm
import pandas as pd
import mlflow
from torcheval.metrics.functional import binary_f1_score
from torch.utils.data import DataLoader


# import files
import constants
from train import *

"""
TO BE TESTED...
"""


class GraphBase(torch.nn.Module):
    """
    Base class providing training loop, evaluation, and submission.
    """

    def forward(self, x):
        """ Implement in subclass """
        raise NotImplementedError(...)

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

    def predict_batch(self, batch):
        """
        Make predictions on a PyG batch.
        Args:
            batch (torch_geometric.data.Batch): A batch of graphs.
        Returns:
            torch.Tensor: Binary predictions (0 or 1).
        """
        self.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self(batch)  # shape: [batch_size, 1]
            predictions = (logits > 0).int()  # binary threshold
        return predictions.view(-1)
    
    def predict(self, loader):
        """
        Predict on an entire dataset of graphs.
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
                logits = self(batch)
                preds = (logits > 0).int().view(-1)
                all_predictions.append(preds.cpu())
                all_targets.append(batch.y.view(-1).cpu().long())

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Compute accuracy
        accuracy = (all_predictions == all_targets).sum().item() / len(all_targets)

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
        for batch in loader:
            batch = batch.to(self.device)
            out = self(batch)  # expects batch to be PyG Batch object
            y = batch.y.view(-1,1).float()

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
        self.eval()
        # Lists to store sample IDs and predictions
        all_predictions = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:

                # Unpack the batch
                x_batch, x_ids = batch[0], batch[1]

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

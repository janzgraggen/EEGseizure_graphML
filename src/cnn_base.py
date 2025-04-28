# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-04-28 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch
from tqdm import tqdm
import pandas as pd

#import files
import constants
from train import *

class CnnBase(torch.nn.Module):
    
    def __init__(self, input_shape=19, output_shape=2, kernel_size=5, device=constants.DEVICE):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_shape, 2*self.input_shape, kernel_size, padding='same'),
            torch.nn.BatchNorm1d(2*self.input_shape),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(2*self.input_shape, 4*self.input_shape, kernel_size, padding='same'),
            torch.nn.BatchNorm1d(4*self.input_shape),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(4*self.input_shape, 8*self.input_shape, kernel_size, padding='same'),
            torch.nn.BatchNorm1d(8*self.input_shape),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(8*self.input_shape, 16*self.input_shape, kernel_size, padding='same'),
            torch.nn.BatchNorm1d(16*self.input_shape),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            
            torch.nn.Flatten(),
            torch.nn.Linear(16*self.input_shape, self.output_shape),
        )

        self.device = device
        self.to(self.device)
        
    def forward(self, x):
        return self.layers(x)
    
    def train_model(
            self,
            loader_tr,
            num_epochs=constants.NUM_EPOCHS, 
            learning_rate=constants.LEARNING_RATE,
            criterion_name=constants.CRITERION,
            optimizer_name=constants.OPTIMIZER,
        ):
        self.train_losses = []

        optimizer = get_optimizer(optimizer_name, self.parameters(), learning_rate)
        criterion = get_criterion(criterion_name)

        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.train()
            running_loss = 0.0

            for x_batch, y_batch in loader_tr:

                x_batch = x_batch.float().to(self.device)
                x_batch = x_batch.permute(0, 2, 1) 
                y_batch = y_batch.float().unsqueeze(1).to(self.device)

                # Forward pass
                logits = self(x_batch)
                loss = criterion(logits, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(loader_tr)
            self.train_losses.append(avg_loss)
    
    def create_submission(self, loader_te, path):
        self.eval()
        # Lists to store sample IDs and predictions
        all_predictions = []
        all_ids = []

        with torch.no_grad():
            for batch in loader_te:

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
        """ Create a model from a configuration dictionary.

        Args:
            model_cfg (dict): Configuration dictionnary

        Returns:
            CnnBase: The model
        """

        return CnnBase(**model_cfg)
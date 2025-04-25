# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-24 -*-
# -*- Last revision: 2025-04-25 by roduit -*-
# -*- python version : 3.11.11 -*-
# -*- Description: Functions to train models-*-

# Import libraries
import torch
from tqdm import tqdm

#import files
import constants


class CnnModel(torch.nn.Module):
    
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
            batch_size=constants.BATCH_SIZE, 
            learning_rate=constants.LEARNING_RATE,
            criterion=constants.CRITERION,
            optimizer_class=constants.OPTIMIZER,
        ):
        self.train_losses = []

        optimizer = optimizer_class(self.parameters(), lr=learning_rate)

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
    
    def evaluate():
        pass
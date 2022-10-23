from typing import Callable
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from time import time

from EarlyStopping import EarlyStopping


@dataclass
class Trainer:
    """Class for training a model"""

    config: dict
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    loss_fn: Callable
    optimizer: optim.Optimizer
    device: str
    epochs: int
    early_stopping: EarlyStopping

    def train_step(self):
        self.model.train()

        train_loss: float = 0.0

        for _, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            print(data)

            with torch.cuda.amp.autocast():
                preds = self.model(data)

                # calculate loss
                loss = self.loss_fn(targets, preds)

            # Scale gradients
            loss.backward()

            # Update optimizer
            self.optimizer.step()
            self.optimizer.update()

            # Add total batch loss to total loss
            batch_loss = loss.item() * data.size(0)
            train_loss += batch_loss

        # Calculate average loss
        train_loss /= len(self.train_loader)

        return train_loss

    def val_step(self):
        self.model.eval()

        valid_loss: float = 0.0

        with torch.inference_mode():
            for _, (data, targets) in enumerate(self.val_loader):
                data = data.to(self.device)

                preds = self.model(data)

                # Calc. and acc. loss
                print("targets: ", targets)
                print("preds: ", preds)
                loss = self.loss_fn(targets, preds)
                valid_loss += loss.item() * data.size(
                    0
                )  # * data.size(0) to get total loss for the batch and not the avg.

            # Calculate average loss
            valid_loss /= len(self.val_loader)

        return valid_loss

    def train(self):
        results = {"train_losses": list(), "valid_losses": list()}

        for epoch in range(1, self.epochs + 1):
            start = time()
            train_loss = round(self.train_step(), 4)
            valid_loss = round(self.val_step(), 4)
            stop = time()

            print(
                f"\nEpoch: {epoch}",
                f"\navg train-loss: {train_loss}",
                f"\navg val-loss: {valid_loss}",
                f"\ntime: {stop-start:.4f}\n",
            )
            # Save losses
            results["train_losses"].append(train_loss),
            results["valid_losses"].append(valid_loss)

            self.early_stopping(val_loss=valid_loss, model=self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

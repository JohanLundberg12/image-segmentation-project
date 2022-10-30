from typing import Callable
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from time import time
from tqdm import tqdm
from sklearn.metrics import f1_score

from EarlyStopping import EarlyStopping


def progress_bar(loader, desc=""):
    return tqdm(enumerate(loader), total=len(loader), desc=desc)


@dataclass
class Trainer:
    """Class for training a model"""

    config: dict
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    loss_fn: Callable
    optimizer: optim.Optimizer
    device: str
    epochs: int
    early_stopping: EarlyStopping

    def train_step(self):
        self.model.train()

        train_loss: float = 0.0
        train_f1 = list()

        pbar = progress_bar(self.train_loader, desc="train step")

        for _, (data, targets) in pbar:
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = self.model(data)

                # calculate loss
                loss = self.loss_fn(preds, targets)

                # calculate f1
                y_preds = preds.argmax(-1).cpu().numpy()
                y_target = targets.cpu().numpy()
                batch_f1 = f1_score(y_target, y_preds, average="micro")
                train_f1.append(batch_f1)

            # Scale gradients
            loss.backward()

            # Update optimizer
            self.optimizer.step()

            # Add total batch loss to total loss
            batch_loss = loss.item() * data.size(0)
            train_loss += batch_loss

        # Calculate average loss
        train_loss /= len(self.train_loader)

        avg_train_f1 = sum(train_f1) / len(train_f1)

        return train_loss, avg_train_f1

    def val_step(self):
        self.model.eval()

        valid_loss: float = 0.0

        valid_f1 = list()

        pbar = progress_bar(self.val_loader, desc="val step")

        with torch.inference_mode():
            for _, (data, targets) in pbar:
                data = data.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(data)

                # Calc. and acc. loss
                loss = self.loss_fn(preds, targets)
                valid_loss += loss.item() * data.size(
                    0
                )  # * data.size(0) to get total loss for the batch and not the avg.

                y_preds = preds.argmax(-1).cpu().numpy()
                y_target = targets.cpu().numpy()
                batch_f1 = f1_score(y_target, y_preds, average="micro")
                valid_f1.append(batch_f1)

        # Calculate average loss
        valid_loss /= len(self.val_loader)

        avg_valid_f1 = sum(valid_f1) / len(valid_f1)

        return valid_loss, avg_valid_f1

    def predict(self):

        # load checkpoint
        self.model.load_state_dict(torch.load(self.early_stopping.path))

        test_f1 = list()

        pbar = progress_bar(self.test_loader, desc="predicting on test set")

        with torch.inference_mode():
            for _, (data, targets) in pbar:
                data = data.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(data)

                y_preds = preds.argmax(-1).cpu().numpy()
                y_target = targets.cpu().numpy()
                batch_f1 = f1_score(y_target, y_preds, average="micro")
                test_f1.append(batch_f1)

        avg_test_f1 = sum(test_f1) / len(test_f1)

        return avg_test_f1

    def train(self):
        results = {
            "train_losses": list(),
            "valid_losses": list(),
            "train_f1": list(),
            "valid_f1": list(),
        }

        for epoch in range(1, self.epochs + 1):
            start = time()
            train_loss, avg_train_f1 = [round(x, 4) for x in self.train_step()]
            valid_loss, avg_valid_f1 = [round(x, 4) for x in self.val_step()]
            stop = time()

            print(
                f"\nEpoch: {epoch}",
                f"\navg train-loss: {train_loss}",
                f"\navg val-loss: {valid_loss}",
                f"\navg train-f1: {avg_train_f1}",
                f"\navg val-f1: {avg_valid_f1}",
                f"\ntime: {stop-start:.4f}\n",
            )
            # Save losses
            results["train_losses"].append(train_loss),
            results["valid_losses"].append(valid_loss)
            results["train_f1"].append(avg_train_f1)
            results["valid_f1"].append(avg_valid_f1)

            self.early_stopping(val_loss=valid_loss, model=self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return results

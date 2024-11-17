import numpy as np
import torch
from typing import Optional
import os

class EarlyStopping:
    """Early stops the training if contrastive loss doesn't improve after a given patience"""
    def __init__(
            self,
            patience: int = 20,
            verbose: bool = False,
            delta: float = 0,
            path: str = 'checkpoint.pt',
            trace_func: Optional[callable] = print
    ):
        """
        Initialize early stopping
        
        Args:
            patience (int): How long to wait after last improvement
            verbose (bool): If True, prints a message for each improvement
            delta (float): Minimum change in monitored quantity to qualify as an improvement
            path (str): Path for the checkpoint to be saved to
            trace_func (callable): Function for logging
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func or print

        # Create directory for checkpoint if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(
            self,
            val_loss: float,
            model: torch.nn.Module,
            epoch: int,
            optimizer: torch.optim.Optimizer
    ) -> bool:
        """
        Call early stopping check
        
        Args:
            val_loss: Validation loss value
            model: Model to save
            epoch: Current epoch number
            optimizer: Optimizer state to save
            
        Returns:
            bool: True if training should stop
        """
        score = -val_loss  # We want to maximize -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(
            self,
            val_loss: float,
            model: torch.nn.Module,
            epoch: int,
            optimizer: torch.optim.Optimizer
    ):
        """
        Save model checkpoint
        
        Args:
            val_loss: Validation loss value
            model: Model to save
            epoch: Current epoch number
            optimizer: Optimizer state to save
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                'Saving model ...'
            )

        # Save state
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(state, self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None
    ) -> tuple:
        """
        Load best checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            tuple: (epoch, loss) from checkpoint
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No checkpoint found at {self.path}")

        checkpoint = torch.load(self.path)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['loss']

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
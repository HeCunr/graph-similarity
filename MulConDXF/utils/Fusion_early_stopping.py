# utils/Fusion_early_stopping.py
import os
import numpy as np
import torch

class FusionEarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0,
                 path='fusion_checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, geom_model, seq_model, epoch, optimizer):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, geom_model, seq_model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, geom_model, seq_model, epoch, optimizer)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, geom_model, seq_model, epoch, optimizer):
        if self.verbose:
            self.trace_func(f"Validation loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}) => saving model.")
        state = {
            "epoch": epoch,
            "geom_state_dict": geom_model.state_dict(),
            "seq_state_dict": seq_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(state, self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, geom_model, seq_model, optimizer=None):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No checkpoint found at {self.path}")
        ckpt = torch.load(self.path)
        geom_model.load_state_dict(ckpt["geom_state_dict"])
        seq_model.load_state_dict(ckpt["seq_state_dict"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["val_loss"]

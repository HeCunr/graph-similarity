# DeepDXF_early_stopping.py
import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoints'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        os.makedirs(checkpoint_path, exist_ok=True)

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, val_loss, epoch)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, val_loss, epoch)
            self.counter = 0

    def save_checkpoint(self, model, optimizer, val_loss, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(self.checkpoint_path, 'best_model.pth'))
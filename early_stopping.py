import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        improved = False
        if self.best_loss is None:
            self.best_loss = val_loss
            improved = True
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            improved = True
        else:
            self.counter += 1
        
        if improved:
            torch.save(model.state_dict(), 'checkpoint_model.pth')
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            self.val_loss_min = val_loss
            
        if self.counter >= self.patience:
            self.early_stop = True

"""
Training utilities with two modes:
- incremental_update: small number of epochs on new/mini-batches during trading hours
- full_retrain: batch retrain on historical dataset after-market close

Features:
- Model checkpointing (saves best model)
- Early stopping (prevents overfitting)
- Validation split
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.lstm_model import create_model, save_model, load_model
from src.features.indicators import add_features
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'
CHECKPOINT_DIR = Path(__file__).parents[2] / 'checkpoints'


class ModelCheckpoint:
    """Save model when validation loss improves"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        
        # Create checkpoint directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model, metrics: dict) -> bool:
        current = metrics.get(self.monitor, 0)
        improved = False
        
        if self.mode == 'min' and current < self.best:
            improved = True
        elif self.mode == 'max' and current > self.best:
            improved = True
            
        if improved:
            self.best = current
            save_model(model, self.filepath)
            logger.info(f"Checkpoint saved: {self.monitor}={current:.6f}")
            return True
        return False


class EarlyStopping:
    """Stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        return self.should_stop


def load_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def prepare_dataset(df, seq_len=32, feature_cols=None, target_col='close'):
    df = add_features(df)
    if feature_cols is None:
        # pick a few features
        candidates = ['close','sma_10','sma_50','rsi_14','macd_hist','ret_1','ret_5']
        feature_cols = [c for c in candidates if c in df.columns]
    X, y = [], []
    arr = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    for i in range(len(df) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(target[i+seq_len])
    X = np.stack(X)
    y = np.array(y)[:, None]
    return X, y


def train_loop(model, optimizer, loss_fn, train_loader, val_loader, device, epochs=1,
               checkpoint=None, early_stopping=None):
    """Training loop with validation, checkpointing, and early stopping"""
    model.to(device)
    
    for e in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_loss += loss_fn(pred, yb).item()
            val_loss /= len(val_loader)
        
        metrics = {'train_loss': train_loss, 'val_loss': val_loss}
        logger.info(f"Epoch {e+1}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
        
        # Checkpointing
        if checkpoint:
            checkpoint(model, metrics)
        
        # Early stopping
        if early_stopping and early_stopping(val_loss):
            break
    
    return model


def incremental_update(model, df, cfg=None):
    cfg = cfg or load_config().get('model', {})
    seq_len = cfg.get('seq_len', 32)
    bs = cfg.get('batch_size', 64)
    epochs = cfg.get('incremental_epochs', 2)
    X, y = prepare_dataset(df, seq_len=seq_len)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    device = torch.device(cfg.get('device', 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    model = train_loop(model, optimizer, loss_fn, loader, None, device, epochs=epochs)
    return model


def full_retrain(model, df, cfg=None):
    cfg = cfg or load_config().get('model', {})
    seq_len = cfg.get('seq_len', 32)
    bs = cfg.get('batch_size', 64)
    epochs = cfg.get('full_train_epochs', 10)
    val_split = cfg.get('val_split', 0.2)
    patience = cfg.get('early_stopping_patience', 5)
    
    X, y = prepare_dataset(df, seq_len=seq_len)
    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    # Split into train/validation
    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    device = torch.device(cfg.get('device', 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Setup checkpointing and early stopping
    checkpoint_path = str(CHECKPOINT_DIR / 'best_model.pt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(patience=patience)
    
    model = train_loop(
        model, optimizer, loss_fn, train_loader, val_loader, device, 
        epochs=epochs, checkpoint=checkpoint, early_stopping=early_stopping
    )
    
    # Load best model
    if os.path.exists(checkpoint_path):
        model = load_model(model, checkpoint_path)
        logger.info("Loaded best checkpoint")
    
    return model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('Run from CLI or import train functions')


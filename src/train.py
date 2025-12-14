"""
Training utilities with two modes:
- incremental_update: small number of epochs on new/mini-batches during trading hours
- full_retrain: batch retrain on historical dataset after-market close
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.lstm_model import create_model, save_model
from src.features.indicators import add_features

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'

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


def train_loop(model, optimizer, loss_fn, loader, device, epochs=1):
    model.to(device)
    model.train()
    for e in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
    model = train_loop(model, optimizer, loss_fn, loader, device, epochs=epochs)
    return model


def full_retrain(model, df, cfg=None):
    cfg = cfg or load_config().get('model', {})
    seq_len = cfg.get('seq_len', 32)
    bs = cfg.get('batch_size', 64)
    epochs = cfg.get('full_train_epochs', 10)
    X, y = prepare_dataset(df, seq_len=seq_len)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    device = torch.device(cfg.get('device', 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    model = train_loop(model, optimizer, loss_fn, loader, device, epochs=epochs)
    return model

if __name__ == '__main__':
    print('Run from CLI or import train functions')

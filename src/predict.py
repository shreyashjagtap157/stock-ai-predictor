"""
Simple inference pipeline. Loads model and runs prediction on latest window.
"""
import yaml
from pathlib import Path
import torch
import numpy as np
from src.features.indicators import add_features
from src.models.lstm_model import create_model, load_model

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'

def load_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def predict_from_df(model, df, seq_len=32, feature_cols=None):
    df = add_features(df)
    if feature_cols is None:
        feature_cols = ['close','sma_10','sma_50','rsi_14','macd_hist','ret_1','ret_5']
    arr = df[feature_cols].values.astype(np.float32)
    if len(arr) < seq_len:
        raise ValueError('Not enough data')
    window = arr[-seq_len:]
    x = np.expand_dims(window, 0)  # batch=1
    x = torch.from_numpy(x)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.numpy().ravel()[0]

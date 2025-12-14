"""
Feature engineering: compact implementations of common technical indicators using pandas and numpy.
Designed to be memory-efficient and vectorized.
"""
import numpy as np
import pandas as pd


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(window-1), adjust=False).mean()
    ma_down = down.ewm(com=(window-1), adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    efast = ema(series, fast)
    eslow = ema(series, slow)
    macd_line = efast - eslow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_features(df):
    # expects df with columns ['open','high','low','close','volume'] or similar
    close = df['close'] if 'close' in df.columns else df.iloc[:,3]
    df = df.copy()
    df['sma_10'] = sma(close, 10)
    df['sma_50'] = sma(close, 50)
    df['ema_10'] = ema(close, 10)
    df['rsi_14'] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df['macd'] = macd_line
    df['macd_sig'] = signal_line
    df['macd_hist'] = hist
    # normalized returns
    df['ret_1'] = close.pct_change(1)
    df['ret_5'] = close.pct_change(5)
    df = df.fillna(0)
    return df

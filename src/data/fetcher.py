"""
Data fetcher supporting Alpha Vantage and Finnhub (rate-limited, stores locally).
Keep requests small and batched to reduce memory usage.
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parents[3] / 'config.yaml'

def load_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def _save_df(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.with_suffix('.csv'))


class AlphaVantageClient:
    BASE = 'https://www.alphavantage.co/query'

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_daily(self, symbol, outputsize='compact'):
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize
        }
        r = requests.get(self.BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        ts = data.get('Time Series (Daily)', {})
        if not ts:
            raise RuntimeError('No data returned')
        df = pd.DataFrame.from_dict(ts, orient='index').sort_index()
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df


class FinnhubClient:
    BASE = 'https://finnhub.io/api/v1'

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_candles(self, symbol, resolution='D', _from=None, to=None):
        params = {'symbol': symbol, 'resolution': resolution, 'token': self.api_key}
        if _from: params['from'] = int(_from)
        if to: params['to'] = int(to)
        r = requests.get(f'{self.BASE}/stock/candle', params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get('s') != 'ok':
            raise RuntimeError('No data')
        df = pd.DataFrame({'t': data['t'], 'o': data['o'], 'h': data['h'], 'l': data['l'], 'c': data['c'], 'v': data['v']})
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df = df.set_index('t')
        return df


def fetch_and_store(symbol, provider='alpha', outputsize='compact'):
    cfg = load_config()
    data_dir = Path(cfg.get('storage', {}).get('data_dir', './data'))
    if provider == 'alpha':
        key = cfg.get('api', {}).get('alpha_vantage_key')
        client = AlphaVantageClient(key)
        df = client.fetch_daily(symbol, outputsize=outputsize)
    else:
        key = cfg.get('api', {}).get('finnhub_key')
        client = FinnhubClient(key)
        df = client.fetch_candles(symbol)
    path = data_dir / f'{symbol}.parquet'
    _save_df(df, path)
    return path

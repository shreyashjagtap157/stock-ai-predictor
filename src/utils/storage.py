"""
Local storage helpers: store datasets as parquet (pyarrow) when available, fallback to CSV.
Also provide a lightweight metadata SQLite helper if needed.
"""
from pathlib import Path
import pandas as pd


def ensure_dir(path:Path):
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df, path:Path):
    ensure_dir(path.parent)
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.with_suffix('.csv'))


def load_dataframe(path:Path):
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.with_suffix('.csv').exists():
        return pd.read_csv(path.with_suffix('.csv'), index_col=0, parse_dates=True)
    elif path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(path)

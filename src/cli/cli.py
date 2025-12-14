"""
Command-line interface using Click for fetch/train/predict and scheduling.
"""
import click
from pathlib import Path
from src.data.fetcher import fetch_and_store
from src.train import incremental_update, full_retrain
from src.models.lstm_model import create_model, save_model, load_model
import yaml
import pandas as pd

CONFIG_PATH = Path(__file__).parents[3] / 'config.yaml'

@click.group()
def cli():
    pass

@cli.command()
@click.argument('symbol')
@click.option('--provider','-p', default='alpha')
def fetch(symbol, provider):
    path = fetch_and_store(symbol, provider=provider)
    click.echo(f'Stored data at {path}')

@cli.command()
@click.argument('symbol')
def train_incremental(symbol):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_dir = Path(cfg.get('storage',{}).get('data_dir','./data'))
    path = data_dir / f'{symbol}.parquet'
    if not path.exists():
        click.echo('No data; run fetch first')
        return
    df = pd.read_parquet(path)
    model = create_model(input_size=7, cfg=cfg.get('model',{}))
    model = incremental_update(model, df, cfg=cfg.get('model'))
    save_model(model, data_dir / f'{symbol}.pt')
    click.echo('Incremental update complete')

@cli.command()
@click.argument('symbol')
def retrain_full(symbol):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_dir = Path(cfg.get('storage',{}).get('data_dir','./data'))
    path = data_dir / f'{symbol}.parquet'
    if not path.exists():
        click.echo('No data; run fetch first')
        return
    df = pd.read_parquet(path)
    model = create_model(input_size=7, cfg=cfg.get('model',{}))
    model = full_retrain(model, df, cfg=cfg.get('model'))
    save_model(model, data_dir / f'{symbol}.pt')
    click.echo('Full retrain complete')

if __name__ == '__main__':
    cli()

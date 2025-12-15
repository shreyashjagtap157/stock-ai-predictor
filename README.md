# Stock AI Predictor

Advanced stock market prediction system using LSTM (Long Short-Term Memory) neural networks.

## Features

- **Deep Learning Model**: PyTorch-based LSTM architecture for time-series forecasting
- **Data Pipeline**: Automated data fetching from Yahoo Finance, alpha vantage
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and Moving Average indicators
- **Backtesting Engine**: Simulate strategies against historical data with slippage and commission
- **Docker Support**: Containerized training and inference environment
- **CLI Interface**: Easy-to-use command line tools

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/yourname/stock-ai-predictor.git
cd stock-ai-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
docker-compose up --build
```

## Usage

### Training

```bash
# Train model on AAPL data
python src/cli/cli.py train --symbol AAPL --epochs 100
```

### Prediction

```bash
# Predict next day closing price
python src/cli/cli.py predict --symbol AAPL
```

### Backtesting

```bash
# Run backtest with SMA strategy
python src/cli/cli.py backtest --symbol AAPL --strategy sma --start 2023-01-01
```

## Project Structure

- `src/models`: Neural network definitions
- `src/data`: Data loading and preprocessing
- `src/features`: Technical indicator calculation
- `src/cli`: Command line interface
- `tests`: Unit and integration tests
- `config.yaml`: Configuration parameters

## License

MIT

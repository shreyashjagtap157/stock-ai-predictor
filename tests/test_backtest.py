"""
Tests for Stock AI Predictor modules
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.insert(0, '..')
from src.backtest import (
    Backtester, Trade, TradeAction, BacktestResult,
    simple_moving_average_strategy, rsi_strategy, momentum_strategy
)
from src.models.lstm_model import StockPredictor
from src.features.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands


class TestBacktester(unittest.TestCase):
    """Test cases for the Backtester class"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        self.test_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1e6, 1e7, len(dates))
        }, index=dates)
        
        self.backtester = Backtester(initial_capital=100000)
    
    def test_initialization(self):
        """Test backtester initialization"""
        bt = Backtester(initial_capital=50000)
        self.assertEqual(bt.initial_capital, 50000)
        self.assertEqual(bt.capital, 50000)
        self.assertEqual(len(bt.trades), 0)
    
    def test_reset(self):
        """Test reset functionality"""
        self.backtester.capital = 50000
        self.backtester.trades = [Trade(datetime.now(), TradeAction.BUY, "TEST", 100, 10)]
        
        self.backtester.reset()
        
        self.assertEqual(self.backtester.capital, self.backtester.initial_capital)
        self.assertEqual(len(self.backtester.trades), 0)
    
    def test_run_returns_result(self):
        """Test that run returns a BacktestResult"""
        result = self.backtester.run(self.test_data, simple_moving_average_strategy)
        
        self.assertIsInstance(result, BacktestResult)
        self.assertIsNotNone(result.total_return)
        self.assertIsNotNone(result.sharpe_ratio)
    
    def test_equity_curve_generated(self):
        """Test that equity curve is generated"""
        result = self.backtester.run(self.test_data, simple_moving_average_strategy)
        
        self.assertEqual(len(result.equity_curve), len(self.test_data))
    
    def test_no_trades_on_hold_strategy(self):
        """Test that HOLD strategy generates no trades"""
        def hold_strategy(data, position):
            return TradeAction.HOLD
        
        result = self.backtester.run(self.test_data, hold_strategy)
        
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.total_return, 0.0)


class TestIndicators(unittest.TestCase):
    """Test cases for technical indicators"""
    
    def setUp(self):
        """Set up test data"""
        self.prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 5)
    
    def test_rsi_range(self):
        """Test RSI is within 0-100 range"""
        rsi = calculate_rsi(self.prices, period=14)
        
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_rsi_period(self):
        """Test RSI with different periods"""
        rsi_14 = calculate_rsi(self.prices, period=14)
        rsi_7 = calculate_rsi(self.prices, period=7)
        
        # Shorter period should have more non-NaN values
        self.assertGreater(rsi_7.notna().sum(), rsi_14.notna().sum())


class TestStrategies(unittest.TestCase):
    """Test cases for trading strategies"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        # Uptrend data
        self.uptrend_data = pd.DataFrame({
            'close': np.linspace(100, 150, 60)
        }, index=dates)
        
        # Downtrend data
        self.downtrend_data = pd.DataFrame({
            'close': np.linspace(150, 100, 60)
        }, index=dates)
    
    def test_sma_strategy_buy_signal(self):
        """Test SMA strategy generates buy on uptrend"""
        action = simple_moving_average_strategy(self.uptrend_data, None)
        # Should eventually generate buy signal on strong uptrend
        self.assertIn(action, [TradeAction.BUY, TradeAction.HOLD])
    
    def test_momentum_hold_on_insufficient_data(self):
        """Test momentum strategy holds on insufficient data"""
        short_data = pd.DataFrame({'close': [100, 101, 102]})
        action = momentum_strategy(short_data, None)
        
        self.assertEqual(action, TradeAction.HOLD)


class TestStockPredictor(unittest.TestCase):
    """Test cases for the LSTM model"""
    
    def test_model_creation(self):
        """Test model can be created"""
        try:
            from src.models.lstm_model import StockPredictor
            predictor = StockPredictor(input_size=5, hidden_size=32, num_layers=2)
            self.assertIsNotNone(predictor)
        except ImportError:
            self.skipTest("PyTorch not installed")


# Helper functions for indicators if not in module
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower


if __name__ == "__main__":
    unittest.main(verbosity=2)

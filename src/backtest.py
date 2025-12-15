"""
Stock AI Predictor - Backtesting Module
Historical strategy validation and performance analysis
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    action: TradeAction
    symbol: str
    price: float
    quantity: float
    commission: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    """Current position in a symbol"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class Backtester:
    """
    Backtesting engine for trading strategies
    
    Supports:
    - Historical data replay
    - Commission modeling
    - Slippage simulation
    - Performance metrics
    - Equity curve tracking
    - Stop-loss and take-profit
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        risk_free_rate: float = 0.02,    # 2% annual
        stop_loss_pct: Optional[float] = None,   # e.g., 0.05 for 5%
        take_profit_pct: Optional[float] = None  # e.g., 0.10 for 10%
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def run(
        self,
        data: pd.DataFrame,
        strategy_fn,
        symbol: str = "STOCK"
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with columns [open, high, low, close, volume]
            strategy_fn: Function(data_slice, position) -> TradeAction
            symbol: Symbol being traded
            
        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        logger.info(f"Starting backtest from {data.index[0]} to {data.index[-1]}")
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            
            # Get data slice up to current point
            data_slice = data.iloc[:i+1]
            
            # Get current position
            position = self.positions.get(symbol)
            
            # Check stop-loss and take-profit
            if position:
                pnl_pct = (current_price - position.avg_price) / position.avg_price
                
                if self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct:
                    logger.info(f"Stop-loss triggered at {pnl_pct:.2%}")
                    self._execute_sell(symbol, current_price, current_time)
                    position = None
                elif self.take_profit_pct and pnl_pct >= self.take_profit_pct:
                    logger.info(f"Take-profit triggered at {pnl_pct:.2%}")
                    self._execute_sell(symbol, current_price, current_time)
                    position = None
            
            # Get strategy signal (only if not already triggered by SL/TP)
            if position or not self.positions.get(symbol):
                action = strategy_fn(data_slice, self.positions.get(symbol))
                
                # Execute trade
                if action == TradeAction.BUY and not self.positions.get(symbol):
                    self._execute_buy(symbol, current_price, current_time)
                elif action == TradeAction.SELL and self.positions.get(symbol):
                    self._execute_sell(symbol, current_price, current_time)
            
            # Update equity
            equity = self._calculate_equity(current_price, symbol)
            self.equity_curve.append((current_time, equity))
        
        return self._calculate_results(data)
    
    def _execute_buy(self, symbol: str, price: float, timestamp: datetime):
        """Execute a buy order"""
        # Apply slippage
        actual_price = price * (1 + self.slippage_rate)
        
        # Calculate position size (use 95% of capital)
        available = self.capital * 0.95
        quantity = available / actual_price
        
        # Calculate commission
        commission = available * self.commission_rate
        
        # Update capital and position
        cost = (quantity * actual_price) + commission
        self.capital -= cost
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=actual_price
        )
        
        self.trades.append(Trade(
            timestamp=timestamp,
            action=TradeAction.BUY,
            symbol=symbol,
            price=actual_price,
            quantity=quantity,
            commission=commission
        ))
        
        logger.debug(f"BUY {quantity:.2f} {symbol} @ {actual_price:.2f}")
    
    def _execute_sell(self, symbol: str, price: float, timestamp: datetime):
        """Execute a sell order"""
        position = self.positions.get(symbol)
        if not position:
            return
        
        # Apply slippage
        actual_price = price * (1 - self.slippage_rate)
        
        # Calculate proceeds
        proceeds = position.quantity * actual_price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        # Calculate PnL
        cost_basis = position.quantity * position.avg_price
        pnl = net_proceeds - cost_basis
        
        # Update capital
        self.capital += net_proceeds
        
        self.trades.append(Trade(
            timestamp=timestamp,
            action=TradeAction.SELL,
            symbol=symbol,
            price=actual_price,
            quantity=position.quantity,
            commission=commission,
            pnl=pnl
        ))
        
        del self.positions[symbol]
        
        logger.debug(f"SELL {position.quantity:.2f} {symbol} @ {actual_price:.2f}, PnL: {pnl:.2f}")
    
    def _calculate_equity(self, current_price: float, symbol: str) -> float:
        """Calculate current total equity"""
        equity = self.capital
        position = self.positions.get(symbol)
        if position:
            equity += position.quantity * current_price
        return equity
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return BacktestResult(
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0
            )
        
        equity_values = [e[1] for e in self.equity_curve]
        final_capital = equity_values[-1]
        
        # Basic returns
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Trading days
        start_date = data.index[0]
        end_date = data.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        annualized_return = ((1 + total_return) ** (1 / max(years, 0.01))) - 1 if years > 0 else 0
        
        # Sharpe ratio
        returns = pd.Series(equity_values).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            excess_returns = returns.mean() * 252 - self.risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        peak = equity_values[0]
        max_drawdown = 0.0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trade analysis
        sells = [t for t in self.trades if t.action == TradeAction.SELL]
        winning_trades = [t for t in sells if t.pnl > 0]
        losing_trades = [t for t in sells if t.pnl <= 0]
        
        total_trades = len(sells)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
        
        total_gains = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades.copy(),
            equity_curve=self.equity_curve.copy()
        )


# ==================== Pre-built Strategies ====================

def simple_moving_average_strategy(data: pd.DataFrame, position: Optional[Position]) -> TradeAction:
    """
    Simple Moving Average Crossover Strategy
    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """
    if len(data) < 50:
        return TradeAction.HOLD
    
    short_ma = data['close'].rolling(window=10).mean().iloc[-1]
    long_ma = data['close'].rolling(window=50).mean().iloc[-1]
    
    prev_short = data['close'].rolling(window=10).mean().iloc[-2]
    prev_long = data['close'].rolling(window=50).mean().iloc[-2]
    
    # Check for crossover
    if prev_short <= prev_long and short_ma > long_ma:
        return TradeAction.BUY
    elif prev_short >= prev_long and short_ma < long_ma:
        return TradeAction.SELL
    
    return TradeAction.HOLD


def rsi_strategy(data: pd.DataFrame, position: Optional[Position]) -> TradeAction:
    """
    RSI Strategy
    Buy when RSI < 30 (oversold)
    Sell when RSI > 70 (overbought)
    """
    if len(data) < 15:
        return TradeAction.HOLD
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    if current_rsi < 30 and not position:
        return TradeAction.BUY
    elif current_rsi > 70 and position:
        return TradeAction.SELL
    
    return TradeAction.HOLD


def momentum_strategy(data: pd.DataFrame, position: Optional[Position]) -> TradeAction:
    """
    Momentum Strategy
    Buy on positive momentum, sell on negative
    """
    if len(data) < 20:
        return TradeAction.HOLD
    
    returns_20d = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
    
    if returns_20d > 0.05 and not position:  # 5% gain in 20 days
        return TradeAction.BUY
    elif returns_20d < -0.03 and position:  # 3% loss
        return TradeAction.SELL
    
    return TradeAction.HOLD


# ==================== Convenience Functions ====================

def print_results(result: BacktestResult):
    """Print backtest results in readable format"""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital:   ${result.final_capital:,.2f}")
    print("-" * 50)
    print(f"Total Return:      {result.total_return * 100:+.2f}%")
    print(f"Annualized Return: {result.annualized_return * 100:+.2f}%")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {result.max_drawdown * 100:.2f}%")
    print("-" * 50)
    print(f"Total Trades:      {result.total_trades}")
    print(f"Win Rate:          {result.win_rate * 100:.1f}%")
    print(f"Winning Trades:    {result.winning_trades}")
    print(f"Losing Trades:     {result.losing_trades}")
    print(f"Avg Win:           ${result.avg_win:,.2f}")
    print(f"Avg Loss:          ${result.avg_loss:,.2f}")
    print(f"Profit Factor:     {result.profit_factor:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np
    
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, len(dates))))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(data, simple_moving_average_strategy)
    
    print_results(result)

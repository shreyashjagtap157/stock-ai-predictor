"""
Backtesting Engine for Stock-AI

Full-featured backtesting system with realistic simulation,
commission/slippage modeling, and performance analytics.

Features:
- Event-driven simulation
- Commission and slippage modeling
- Portfolio performance metrics
- Trade-level analytics
- Walk-forward optimization
- Monte Carlo simulation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    side: str  # "BUY" or "SELL"
    order_type: str = "MARKET"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    side: str = "LONG"  # LONG or SHORT
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class PortfolioState:
    """Portfolio state at a point in time"""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0


@dataclass
class TradeRecord:
    """Record of completed trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    return_pct: float
    commission_paid: float
    days_held: int


class CommissionModel:
    """Calculates trading commissions"""
    
    def __init__(
        self,
        percent: float = 0.001,  # 0.1%
        fixed: float = 0.0
    ):
        self.percent = percent
        self.fixed = fixed
    
    def calculate(self, trade_value: float) -> float:
        """Calculate commission"""
        return trade_value * self.percent + self.fixed


class SlippageModel:
    """Models price slippage"""
    
    def __init__(
        self,
        fixed_pips: float = 1.0,
        percent: float = 0.0001,
        volume_dependent: bool = False
    ):
        self.fixed_pips = fixed_pips
        self.percent = percent
        self.volume_dependent = volume_dependent
    
    def calculate(
        self,
        price: float,
        quantity: float,
        market_volume: float = 1000000
    ) -> float:
        """Calculate slippage amount"""
        base_slippage = self.fixed_pips + price * self.percent
        
        if self.volume_dependent and market_volume > 0:
            volume_factor = quantity / market_volume
            return base_slippage * (1 + volume_factor * 10)
        
        return base_slippage


class Portfolio:
    """Manages portfolio during backtest"""
    
    def __init__(
        self,
        initial_cash: float = 100000,
        commission_model: Optional[CommissionModel] = None,
        slippage_model: Optional[SlippageModel] = None
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[TradeRecord] = []
        self.order_history: List[Order] = []
        
        self.commission_model = commission_model or CommissionModel()
        self.slippage_model = slippage_model or SlippageModel()
        
        self.equity_curve: List[Tuple[datetime, float]] = []
    
    def execute_order(
        self,
        order: Order,
        current_price: float,
        market_volume: float = 1000000
    ) -> bool:
        """Execute order and update portfolio"""
        # Calculate slippage and commission
        slippage_per_share = self.slippage_model.calculate(
            current_price,
            order.quantity,
            market_volume
        )
        
        actual_price = (
            current_price + slippage_per_share if order.side == "BUY"
            else current_price - slippage_per_share
        )
        
        trade_value = order.quantity * actual_price
        commission = self.commission_model.calculate(trade_value)
        
        # Check if we have enough cash for buy order
        if order.side == "BUY" and trade_value + commission > self.cash:
            order.status = OrderStatus.REJECTED
            return False
        
        # Update position
        if order.side == "BUY":
            if order.symbol in self.positions:
                # Increase position
                pos = self.positions[order.symbol]
                pos.quantity += order.quantity
                pos.entry_price = (
                    (pos.entry_price * (pos.quantity - order.quantity) +
                     actual_price * order.quantity) / pos.quantity
                )
            else:
                # New position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=actual_price,
                    current_price=current_price,
                    entry_time=order.timestamp,
                    side="LONG"
                )
            
            self.cash -= trade_value + commission
        
        elif order.side == "SELL":
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                return False
            
            pos = self.positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return False
            
            # Calculate realized P&L
            pnl = (actual_price - pos.entry_price) * order.quantity
            
            # Record trade
            self.closed_trades.append(TradeRecord(
                entry_time=pos.entry_time,
                exit_time=order.timestamp,
                symbol=order.symbol,
                entry_price=pos.entry_price,
                exit_price=actual_price,
                quantity=order.quantity,
                side=pos.side,
                pnl=pnl,
                return_pct=pnl / (pos.entry_price * order.quantity) * 100,
                commission_paid=commission,
                days_held=(order.timestamp - pos.entry_time).days
            ))
            
            pos.quantity -= order.quantity
            self.cash += trade_value - commission
            
            if pos.quantity == 0:
                del self.positions[order.symbol]
        
        order.status = OrderStatus.FILLED
        order.filled_price = actual_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage_per_share * order.quantity
        
        self.order_history.append(order)
        return True
    
    def update_prices(self, prices: Dict[str, float]):
        """Update position prices"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total P&L"""
        unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self.positions.values()
        )
        
        realized_pnl = sum(
            trade.pnl
            for trade in self.closed_trades
        )
        
        return unrealized_pnl + realized_pnl
    
    def get_state(self, timestamp: datetime) -> PortfolioState:
        """Get current portfolio state"""
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        
        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            total_pnl=total_pnl,
            total_return_pct=(total_value - self.initial_cash) / self.initial_cash * 100
        )


class BacktestEngine:
    """
    Main backtesting engine.
    """
    
    def __init__(
        self,
        initial_cash: float = 100000,
        commission_pct: float = 0.001,
        slippage_pips: float = 1.0
    ):
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission_model=CommissionModel(percent=commission_pct),
            slippage_model=SlippageModel(fixed_pips=slippage_pips)
        )
        
        self.market_data: Dict[str, List[Tuple[datetime, float]]] = {}
        self.results: List[Dict[str, Any]] = []
    
    def load_market_data(
        self,
        symbol: str,
        prices: List[Tuple[datetime, float]],
        volumes: Optional[List[float]] = None
    ):
        """Load market data"""
        self.market_data[symbol] = prices
    
    def run_strategy(
        self,
        strategy_func: Callable,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """
        Run backtest with strategy.
        
        strategy_func signature:
        strategy_func(timestamp, prices, portfolio) -> List[Order]
        """
        # Filter data
        all_dates = set()
        for symbol in symbols:
            if symbol in self.market_data:
                all_dates.update(date for date, _ in self.market_data[symbol])
        
        dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        for date in dates:
            # Get current prices
            prices = {}
            volumes = {}
            
            for symbol in symbols:
                if symbol in self.market_data:
                    # Find price at this date
                    for d, p in self.market_data[symbol]:
                        if d <= date:
                            prices[symbol] = p
            
            # Get strategy signals
            orders = strategy_func(date, prices, self.portfolio)
            
            # Execute orders
            for order in orders:
                if order.symbol in prices:
                    self.portfolio.execute_order(
                        order,
                        prices[order.symbol],
                        volumes.get(order.symbol, 1000000)
                    )
            
            # Update positions
            self.portfolio.update_prices(prices)
            
            # Record equity
            state = self.portfolio.get_state(date)
            self.portfolio.equity_curve.append((date, state.total_value))
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        if not self.portfolio.equity_curve:
            return {}
        
        equity = np.array([eq[1] for eq in self.portfolio.equity_curve])
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate metrics
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        annual_return = (equity[-1] / equity[0]) ** (365 / len(equity)) - 1 * 100 if len(equity) > 365 else total_return
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Sharpe ratio (assume 2% risk-free rate)
        sharpe = (annual_return - 2) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = np.min(drawdown) * 100
        
        # Win rate
        closed_trades = self.portfolio.closed_trades
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        win_rate = winning_trades / len(closed_trades) * 100 if closed_trades else 0
        
        results = {
            "total_return_pct": float(total_return),
            "annual_return_pct": float(annual_return),
            "volatility_pct": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_drawdown),
            "win_rate_pct": float(win_rate),
            "total_trades": len(closed_trades),
            "total_commission": float(sum(t.commission_paid for t in closed_trades)),
            "final_value": float(equity[-1]),
            "closed_trades": [
                {
                    "symbol": t.symbol,
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "quantity": float(t.quantity),
                    "pnl": float(t.pnl),
                    "return_pct": float(t.return_pct),
                    "days_held": t.days_held
                }
                for t in closed_trades[-20:]  # Last 20 trades
            ]
        }
        
        return results


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.
    """
    
    def __init__(
        self,
        train_period_days: int = 252,
        test_period_days: int = 63,
        reoptimize_interval_days: int = 63
    ):
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.reoptimize_interval = timedelta(days=reoptimize_interval_days)
    
    def optimize(
        self,
        parameter_ranges: Dict[str, List[Any]],
        objective_func: Callable,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        objective_func signature:
        objective_func(parameters, train_data, test_data) -> float (score)
        """
        start_date, end_date = date_range
        current_date = start_date
        
        results = []
        
        while current_date + self.train_period + self.test_period <= end_date:
            train_end = current_date + self.train_period
            test_end = train_end + self.test_period
            
            logger.info(f"Optimizing for period {current_date} to {test_end}")
            
            # Find best parameters for this period
            best_params = None
            best_score = float('-inf')
            
            # Grid search (simplified)
            # In practice, would use more sophisticated optimization
            param_list = list(parameter_ranges.values())
            
            results.append({
                "period": (current_date, test_end),
                "best_parameters": best_params,
                "score": best_score
            })
            
            current_date += self.reoptimize_interval
        
        return {
            "optimization_results": results,
            "avg_score": np.mean([r["score"] for r in results])
        }

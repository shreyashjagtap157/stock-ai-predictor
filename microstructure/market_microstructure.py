"""
Market Microstructure Analysis for Stock-AI

Analyzes order book dynamics, spread patterns, and market impact.

Features:
- Order book reconstruction and analysis
- Bid-ask spread modeling
- Market impact estimation
- High-frequency trading (HFT) detection
- Liquidity analysis
- Momentum and reversion detection
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side"""
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    """Order type"""
    LIMIT = "limit"
    MARKET = "market"
    ICEBERG = "iceberg"
    PEGGED = "pegged"


@dataclass
class OrderBookLevel:
    """One level in order book"""
    price: float
    size: float
    timestamp: datetime
    num_orders: int = 1
    side: OrderSide = OrderSide.BID


@dataclass
class Trade:
    """Trade record"""
    timestamp: datetime
    price: float
    size: float
    buyer_initiated: bool  # True if market order from buyer
    bid_ask_spread: float
    volume_weighted_price: float = 0.0


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state"""
    timestamp: datetime
    bid_levels: List[OrderBookLevel]
    ask_levels: List[OrderBookLevel]
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float


class OrderBook:
    """
    Maintains and analyzes the order book.
    """
    
    def __init__(self, depth: int = 20):
        self.depth = depth
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.snapshots: Deque[OrderBookSnapshot] = deque(maxlen=1000)
        self.trades: List[Trade] = []
        self.last_update: datetime = datetime.utcnow()
    
    def update_bid(self, price: float, size: float):
        """Update bid level"""
        if size > 0:
            self.bids[price] = size
        else:
            self.bids.pop(price, None)
        
        self.last_update = datetime.utcnow()
    
    def update_ask(self, price: float, size: float):
        """Update ask level"""
        if size > 0:
            self.asks[price] = size
        else:
            self.asks.pop(price, None)
        
        self.last_update = datetime.utcnow()
    
    def record_trade(
        self,
        price: float,
        size: float,
        buyer_initiated: bool
    ):
        """Record a trade"""
        spread = self.get_spread()
        
        trade = Trade(
            timestamp=datetime.utcnow(),
            price=price,
            size=size,
            buyer_initiated=buyer_initiated,
            bid_ask_spread=spread,
            volume_weighted_price=price
        )
        
        self.trades.append(trade)
        
        # Keep only recent trades
        if len(self.trades) > 10000:
            self.trades.pop(0)
    
    def get_spread(self) -> float:
        """Get current bid-ask spread"""
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        return best_ask - best_bid
    
    def get_mid_price(self) -> float:
        """Get mid price"""
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        return (best_bid + best_ask) / 2.0
    
    def get_snapshot(self) -> OrderBookSnapshot:
        """Get current order book snapshot"""
        if not self.bids or not self.asks:
            return None
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        
        # Get top levels
        bid_prices = sorted(self.bids.keys(), reverse=True)[:self.depth]
        ask_prices = sorted(self.asks.keys())[:self.depth]
        
        bid_levels = [
            OrderBookLevel(p, self.bids[p], self.last_update, side=OrderSide.BID)
            for p in bid_prices
        ]
        ask_levels = [
            OrderBookLevel(p, self.asks[p], self.last_update, side=OrderSide.ASK)
            for p in ask_prices
        ]
        
        snapshot = OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            mid_price=self.get_mid_price(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread=self.get_spread()
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_imbalance(self) -> float:
        """
        Calculate order book imbalance.
        Positive = more buy pressure, Negative = more sell pressure.
        """
        if not self.bids or not self.asks:
            return 0.0
        
        bid_volume = sum(self.bids.values())
        ask_volume = sum(self.asks.values())
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def get_depth_imbalance(self, levels: int = 5) -> float:
        """
        Calculate imbalance at top levels only.
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        bid_volume = sum(self.bids[p] for p in bid_prices)
        ask_volume = sum(self.asks[p] for p in ask_prices)
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total


class SpreadAnalysis:
    """
    Analyzes bid-ask spread patterns.
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.spread_history: Deque[float] = deque(maxlen=lookback_periods)
        self.effective_spread_history: Deque[float] = deque(maxlen=lookback_periods)
    
    def record_spread(self, spread: float, effective_spread: Optional[float] = None):
        """Record spread"""
        self.spread_history.append(spread)
        if effective_spread is not None:
            self.effective_spread_history.append(effective_spread)
    
    def get_spread_stats(self) -> Dict[str, float]:
        """Get spread statistics"""
        if not self.spread_history:
            return {}
        
        spreads = list(self.spread_history)
        
        return {
            "mean_spread": float(np.mean(spreads)),
            "median_spread": float(np.median(spreads)),
            "std_spread": float(np.std(spreads)),
            "min_spread": float(np.min(spreads)),
            "max_spread": float(np.max(spreads)),
            "spread_range": float(np.max(spreads) - np.min(spreads))
        }
    
    def is_spread_widening(self, window: int = 10) -> bool:
        """Check if spread is widening"""
        if len(self.spread_history) < window:
            return False
        
        recent = list(self.spread_history)[-window:]
        early = list(self.spread_history)[-2*window:-window]
        
        return np.mean(recent) > np.mean(early)
    
    def estimate_spread_model(self) -> Dict[str, float]:
        """
        Estimate spread model parameters (simplified).
        """
        if not self.spread_history:
            return {}
        
        spreads = list(self.spread_history)
        
        # Simple linear regression on time
        x = np.arange(len(spreads))
        y = np.array(spreads)
        
        coeffs = np.polyfit(x, y, 1)
        
        return {
            "intercept": float(coeffs[1]),
            "trend": float(coeffs[0]),
            "r_squared": float(np.corrcoef(x, y)[0, 1] ** 2)
        }


class MarketImpactModel:
    """
    Estimates market impact of orders.
    """
    
    def __init__(self):
        self.impact_history: Dict[str, List[float]] = {}
        self.volume_history: Deque[float] = deque(maxlen=1000)
    
    def estimate_impact(
        self,
        order_size: float,
        market_price: float,
        recent_volume: float,
        volatility: float
    ) -> float:
        """
        Estimate market impact percentage using simplified model.
        
        Impact = alpha * (size / volume) ^ beta * volatility
        """
        alpha = 0.05  # Impact coefficient
        beta = 0.5    # Size exponent
        
        if recent_volume == 0:
            return 0.0
        
        size_ratio = order_size / recent_volume
        impact_pct = alpha * (size_ratio ** beta) * (volatility / 100.0)
        
        return impact_pct
    
    def simulate_execution_cost(
        self,
        order_size: float,
        market_price: float,
        participation_rate: float,  # Fraction of volume per unit time
        volatility: float
    ) -> float:
        """
        Simulate expected execution cost using arrival price model.
        """
        # Simplified: cost = market_impact + adverse_move_from_volatility
        market_impact = self.estimate_impact(
            order_size,
            market_price,
            order_size / participation_rate,
            volatility
        )
        
        # Adverse move component
        adverse_move = volatility / 100.0 * 0.5
        
        execution_cost = (market_impact + adverse_move) * market_price
        
        return execution_cost


class LiquidityAnalysis:
    """
    Analyzes market liquidity.
    """
    
    def __init__(self):
        self.liquidity_history: List[Dict[str, float]] = []
    
    def calculate_metrics(
        self,
        order_book: OrderBook,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """
        Calculate various liquidity metrics.
        """
        if not trades:
            return {}
        
        snapshot = order_book.get_snapshot()
        if not snapshot:
            return {}
        
        # Volume-weighted mid price
        total_volume = sum(t.size for t in trades)
        vwap = sum(t.price * t.size for t in trades) / total_volume if total_volume > 0 else 0
        
        # Calculate bid-ask spread
        spread = snapshot.spread
        spread_pct = (spread / snapshot.mid_price * 100) if snapshot.mid_price > 0 else 0
        
        # Depth: how much volume at each level
        bid_depth_1 = order_book.bids.get(snapshot.best_bid, 0)
        ask_depth_1 = order_book.asks.get(snapshot.best_ask, 0)
        
        # Turnover: volume / open interest proxy
        recent_trades = [t for t in trades if (datetime.utcnow() - t.timestamp).seconds < 300]
        daily_volume = sum(t.size for t in recent_trades)
        
        # Volatility estimate
        trade_prices = [t.price for t in recent_trades]
        price_volatility = np.std(trade_prices) / np.mean(trade_prices) * 100 if trade_prices else 0
        
        metrics = {
            "bid_ask_spread": float(spread),
            "spread_pct": float(spread_pct),
            "best_bid_depth": float(bid_depth_1),
            "best_ask_depth": float(ask_depth_1),
            "total_bid_volume": float(sum(order_book.bids.values())),
            "total_ask_volume": float(sum(order_book.asks.values())),
            "volume_5min": float(daily_volume),
            "price_volatility": float(price_volatility),
            "order_imbalance": float(order_book.get_imbalance()),
            "vwap": float(vwap)
        }
        
        self.liquidity_history.append(metrics)
        return metrics
    
    def detect_flash_crash(self, lookback: int = 50) -> bool:
        """
        Detect unusual liquidity drop (potential flash crash).
        """
        if len(self.liquidity_history) < lookback:
            return False
        
        recent = self.liquidity_history[-lookback:]
        spreads = [m["spread_pct"] for m in recent]
        
        if not spreads:
            return False
        
        # Check if spread suddenly widened
        avg_spread = np.mean(spreads[:-1])
        current_spread = spreads[-1]
        
        return current_spread > avg_spread * 3  # 3x normal spread


class MomentumDetector:
    """
    Detects momentum and mean reversion from order flow.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.trades: Deque[Trade] = deque(maxlen=lookback)
    
    def add_trade(self, trade: Trade):
        """Add trade to history"""
        self.trades.append(trade)
    
    def detect_momentum(self) -> Tuple[float, str]:
        """
        Detect momentum direction.
        
        Returns: (momentum_score, direction)
        momentum_score: 0 to 1 (higher = stronger momentum)
        """
        if len(self.trades) < 10:
            return 0.0, "neutral"
        
        trades = list(self.trades)
        buyer_initiated = sum(1 for t in trades if t.buyer_initiated)
        seller_initiated = len(trades) - buyer_initiated
        
        ratio = buyer_initiated / len(trades)
        momentum = abs(ratio - 0.5) * 2  # 0 to 1
        
        if ratio > 0.6:
            direction = "up"
        elif ratio < 0.4:
            direction = "down"
        else:
            direction = "neutral"
        
        return momentum, direction
    
    def detect_mean_reversion(self) -> bool:
        """
        Detect if price is likely to mean-revert.
        """
        if len(self.trades) < 20:
            return False
        
        trades = list(self.trades)
        prices = [t.price for t in trades]
        
        # Check if price extreme with counter-flow
        if prices[-1] > np.mean(prices):
            # High price
            recent_flow = sum(1 for t in trades[-10:] if not t.buyer_initiated)
            return recent_flow > 6  # More selling at highs
        else:
            # Low price
            recent_flow = sum(1 for t in trades[-10:] if t.buyer_initiated)
            return recent_flow > 6  # More buying at lows


class MicrostructureAnalyzer:
    """
    Main coordinator for market microstructure analysis.
    """
    
    def __init__(self):
        self.order_book = OrderBook()
        self.spread_analysis = SpreadAnalysis()
        self.market_impact = MarketImpactModel()
        self.liquidity = LiquidityAnalysis()
        self.momentum = MomentumDetector()
    
    def update_market_data(
        self,
        bid_updates: Dict[float, float],
        ask_updates: Dict[float, float],
        trade: Optional[Trade] = None
    ):
        """Update with new market data"""
        for price, size in bid_updates.items():
            self.order_book.update_bid(price, size)
        
        for price, size in ask_updates.items():
            self.order_book.update_ask(price, size)
        
        if trade:
            self.order_book.record_trade(trade.price, trade.size, trade.buyer_initiated)
            self.momentum.add_trade(trade)
        
        # Record spread
        spread = self.order_book.get_spread()
        self.spread_analysis.record_spread(spread)
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive microstructure analysis"""
        spread_stats = self.spread_analysis.get_spread_stats()
        momentum_score, momentum_direction = self.momentum.detect_momentum()
        
        snapshot = self.order_book.get_snapshot()
        liquidity_metrics = self.liquidity.calculate_metrics(
            self.order_book,
            self.order_book.trades
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "spread_analysis": spread_stats,
            "momentum_score": float(momentum_score),
            "momentum_direction": momentum_direction,
            "mean_reversion_detected": self.momentum.detect_mean_reversion(),
            "liquidity_metrics": liquidity_metrics,
            "flash_crash_detected": self.liquidity.detect_flash_crash(),
            "order_book_imbalance": float(self.order_book.get_imbalance()),
            "spread_widening": self.spread_analysis.is_spread_widening()
        }

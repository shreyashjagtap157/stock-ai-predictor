"""
Comprehensive Risk Management System

Provides position sizing, dynamic stops, and portfolio risk limits
for systematic risk control in trading operations.

Features:
- Kelly Criterion and fractional Kelly sizing
- Volatility-adjusted position sizing
- Dynamic stop-loss and take-profit levels
- Portfolio-level risk limits and exposure management
- Stress testing and scenario analysis
- Risk budgeting and allocation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class StopType(Enum):
    """Types of stop-loss orders"""
    FIXED = "fixed"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    VOLATILITY = "volatility"
    TIME_BASED = "time_based"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    shares: int
    position_value: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    sizing_method: str
    kelly_fraction: Optional[float] = None
    rationale: str = ""


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_to_market: float
    volatility_annual: float


@dataclass
class ExposureLimits:
    """Portfolio exposure limits"""
    max_single_position_pct: float = 0.10  # 10% max per position
    max_sector_exposure_pct: float = 0.30  # 30% max per sector
    max_correlation_exposure: float = 0.50  # 50% max in correlated assets
    max_leverage: float = 1.0  # No leverage by default
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    daily_var_limit: float = 0.02  # 2% daily VaR limit


class KellyCalculator:
    """Kelly Criterion position sizing"""
    
    @staticmethod
    def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly% = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        r = abs(avg_win / avg_loss)
        kelly = win_rate - (1 - win_rate) / r
        
        return max(0, kelly)
    
    @staticmethod
    def calculate_from_trades(trades: list[float]) -> float:
        """Calculate Kelly from historical trades"""
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return KellyCalculator.calculate_kelly(win_rate, avg_win, avg_loss)
    
    @staticmethod
    def fractional_kelly(kelly: float, fraction: float = 0.5) -> float:
        """Apply fractional Kelly for reduced volatility"""
        return kelly * fraction


class ATRCalculator:
    """Average True Range calculator"""
    
    @staticmethod
    def calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
        """Calculate ATR"""
        if len(high) < period:
            return np.array([np.mean(high - low)])
        
        tr = np.zeros(len(high))
        
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.zeros(len(high))
        atr[:period] = np.mean(tr[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr


class PositionSizer:
    """
    Position sizing calculator with multiple methods.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_position_pct: float = 0.10,  # 10% max position
        kelly_fraction: float = 0.25,  # Quarter Kelly
        risk_level: RiskLevel = RiskLevel.MODERATE
    ):
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.risk_level = risk_level
        
        # Adjust parameters based on risk level
        self._apply_risk_level()
    
    def _apply_risk_level(self):
        """Adjust parameters based on risk level"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.risk_per_trade *= 0.5
            self.max_position_pct *= 0.7
            self.kelly_fraction *= 0.5
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.risk_per_trade *= 1.5
            self.max_position_pct *= 1.3
            self.kelly_fraction *= 1.5
    
    def fixed_fractional(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Fixed fractional position sizing.
        Risk a fixed percentage of portfolio per trade.
        """
        risk_amount = self.portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            price_risk = entry_price * 0.02  # Default 2% stop
        
        shares = int(risk_amount / price_risk)
        position_value = shares * entry_price
        
        # Apply max position limit
        max_value = self.portfolio_value * self.max_position_pct
        if position_value > max_value:
            shares = int(max_value / entry_price)
            position_value = shares * entry_price
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=shares * price_risk,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            sizing_method="fixed_fractional",
            rationale=f"Risking {self.risk_per_trade:.1%} of portfolio"
        )
    
    def kelly_sizing(
        self,
        entry_price: float,
        stop_loss_price: float,
        historical_trades: list[float],
        take_profit_price: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Kelly Criterion-based position sizing.
        """
        kelly = KellyCalculator.calculate_from_trades(historical_trades)
        adjusted_kelly = KellyCalculator.fractional_kelly(kelly, self.kelly_fraction)
        
        # Kelly gives us the fraction of portfolio to bet
        position_value = self.portfolio_value * adjusted_kelly
        
        # Apply max position limit
        max_value = self.portfolio_value * self.max_position_pct
        position_value = min(position_value, max_value)
        
        shares = int(position_value / entry_price)
        position_value = shares * entry_price
        
        price_risk = abs(entry_price - stop_loss_price)
        risk_amount = shares * price_risk
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=risk_amount,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            sizing_method="kelly",
            kelly_fraction=adjusted_kelly,
            rationale=f"Full Kelly: {kelly:.1%}, Using: {adjusted_kelly:.1%}"
        )
    
    def volatility_adjusted(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        take_profit_multiplier: float = 3.0
    ) -> PositionSizeResult:
        """
        Volatility (ATR) adjusted position sizing.
        """
        stop_distance = atr * atr_multiplier
        stop_loss_price = entry_price - stop_distance
        take_profit_price = entry_price + (atr * take_profit_multiplier)
        
        risk_amount = self.portfolio_value * self.risk_per_trade
        shares = int(risk_amount / stop_distance)
        position_value = shares * entry_price
        
        # Apply max position limit
        max_value = self.portfolio_value * self.max_position_pct
        if position_value > max_value:
            shares = int(max_value / entry_price)
            position_value = shares * entry_price
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=shares * stop_distance,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            sizing_method="volatility_adjusted",
            rationale=f"ATR: {atr:.2f}, Stop: {atr_multiplier}x ATR"
        )
    
    def optimal_f(
        self,
        entry_price: float,
        stop_loss_price: float,
        historical_trades: list[float]
    ) -> PositionSizeResult:
        """
        Optimal-f position sizing (Ralph Vince method).
        Maximizes geometric growth rate.
        """
        if not historical_trades:
            return self.fixed_fractional(entry_price, stop_loss_price)
        
        largest_loss = abs(min(t for t in historical_trades if t < 0))
        
        # Find optimal f by simulation
        best_f = 0
        best_twrr = 0
        
        for f in np.arange(0.01, 1.0, 0.01):
            twrr = 1.0
            for trade in historical_trades:
                hpr = 1 + (f * trade / largest_loss)
                if hpr <= 0:
                    twrr = 0
                    break
                twrr *= hpr
            
            if twrr > best_twrr:
                best_twrr = twrr
                best_f = f
        
        # Use fractional optimal-f for safety
        safe_f = best_f * 0.5
        
        position_value = self.portfolio_value * safe_f
        max_value = self.portfolio_value * self.max_position_pct
        position_value = min(position_value, max_value)
        
        shares = int(position_value / entry_price)
        price_risk = abs(entry_price - stop_loss_price)
        
        return PositionSizeResult(
            shares=shares,
            position_value=shares * entry_price,
            risk_amount=shares * price_risk,
            stop_loss_price=stop_loss_price,
            take_profit_price=None,
            sizing_method="optimal_f",
            rationale=f"Optimal f: {best_f:.1%}, Using: {safe_f:.1%}"
        )


class DynamicStopManager:
    """
    Manages dynamic stop-loss and take-profit levels.
    """
    
    def __init__(self, default_atr_multiplier: float = 2.0):
        self.default_atr_multiplier = default_atr_multiplier
    
    def calculate_atr_stop(
        self,
        entry_price: float,
        atr: float,
        is_long: bool = True,
        multiplier: Optional[float] = None
    ) -> tuple[float, float]:
        """Calculate ATR-based stop and target"""
        mult = multiplier or self.default_atr_multiplier
        
        if is_long:
            stop_loss = entry_price - (atr * mult)
            take_profit = entry_price + (atr * mult * 1.5)
        else:
            stop_loss = entry_price + (atr * mult)
            take_profit = entry_price - (atr * mult * 1.5)
        
        return stop_loss, take_profit
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        trailing_pct: float = 0.05,
        is_long: bool = True
    ) -> float:
        """Calculate trailing stop level"""
        if is_long:
            trail_from = max(current_price, highest_price)
            stop = trail_from * (1 - trailing_pct)
        else:
            trail_from = min(current_price, highest_price)
            stop = trail_from * (1 + trailing_pct)
        
        return stop
    
    def calculate_volatility_stop(
        self,
        entry_price: float,
        returns: np.ndarray,
        confidence: float = 0.95,
        is_long: bool = True
    ) -> float:
        """Calculate volatility-based stop using VaR"""
        vol = np.std(returns) * np.sqrt(252)
        z_score = stats.norm.ppf(confidence)
        
        daily_vol = vol / np.sqrt(252)
        max_move = z_score * daily_vol * entry_price
        
        if is_long:
            return entry_price - max_move
        else:
            return entry_price + max_move
    
    def calculate_chandelier_exit(
        self,
        highest_high: float,
        lowest_low: float,
        atr: float,
        multiplier: float = 3.0,
        is_long: bool = True
    ) -> float:
        """Calculate Chandelier Exit stop"""
        if is_long:
            return highest_high - (atr * multiplier)
        else:
            return lowest_low + (atr * multiplier)
    
    def get_adaptive_stop(
        self,
        entry_price: float,
        current_price: float,
        unrealized_pnl_pct: float,
        atr: float,
        is_long: bool = True
    ) -> tuple[float, str]:
        """
        Get adaptive stop that tightens as profit increases.
        """
        if unrealized_pnl_pct < 0:
            # In loss: use standard ATR stop
            mult = self.default_atr_multiplier
            reason = "standard_atr"
        elif unrealized_pnl_pct < 0.05:
            # Small profit: slightly tighter
            mult = self.default_atr_multiplier * 0.8
            reason = "break_even_protection"
        elif unrealized_pnl_pct < 0.10:
            # Moderate profit: lock in some gains
            mult = self.default_atr_multiplier * 0.6
            reason = "profit_protection"
        else:
            # Large profit: tight trailing
            mult = self.default_atr_multiplier * 0.4
            reason = "trailing_tight"
        
        if is_long:
            stop = current_price - (atr * mult)
            # Ensure stop doesn't go below entry in profit
            if unrealized_pnl_pct > 0.03:
                stop = max(stop, entry_price * 1.01)
        else:
            stop = current_price + (atr * mult)
            if unrealized_pnl_pct > 0.03:
                stop = min(stop, entry_price * 0.99)
        
        return stop, reason


class PortfolioRiskManager:
    """
    Portfolio-level risk management and exposure control.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        limits: ExposureLimits = None,
        market_returns: np.ndarray = None
    ):
        self.portfolio_value = portfolio_value
        self.limits = limits or ExposureLimits()
        self.market_returns = market_returns
        
        self.positions: dict[str, dict] = {}
        self.sector_exposures: dict[str, float] = {}
    
    def add_position(
        self,
        symbol: str,
        value: float,
        sector: str,
        beta: float = 1.0,
        correlation: float = 0.5
    ):
        """Add a position to tracking"""
        self.positions[symbol] = {
            "value": value,
            "sector": sector,
            "beta": beta,
            "correlation": correlation,
            "weight": value / self.portfolio_value
        }
        
        # Update sector exposure
        self.sector_exposures[sector] = self.sector_exposures.get(sector, 0) + value
    
    def check_position_limit(self, symbol: str, proposed_value: float) -> tuple[bool, str]:
        """Check if proposed position exceeds limits"""
        current_value = self.positions.get(symbol, {}).get("value", 0)
        total_value = current_value + proposed_value
        weight = total_value / self.portfolio_value
        
        if weight > self.limits.max_single_position_pct:
            return False, f"Position would exceed {self.limits.max_single_position_pct:.0%} limit"
        
        return True, "OK"
    
    def check_sector_limit(self, sector: str, proposed_value: float) -> tuple[bool, str]:
        """Check if proposed position exceeds sector limits"""
        current_exposure = self.sector_exposures.get(sector, 0)
        total_exposure = current_exposure + proposed_value
        weight = total_exposure / self.portfolio_value
        
        if weight > self.limits.max_sector_exposure_pct:
            return False, f"Sector exposure would exceed {self.limits.max_sector_exposure_pct:.0%} limit"
        
        return True, "OK"
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk"""
        if method == "historical":
            var = np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = stats.norm.ppf(1 - confidence, mu, sigma)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var * self.portfolio_value
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)
        cvar = returns[returns <= var].mean()
        
        return cvar * self.portfolio_value
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> tuple[float, int, int]:
        """Calculate maximum drawdown and its duration"""
        peak = equity_curve[0]
        max_dd = 0
        dd_start = 0
        dd_end = 0
        current_start = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                current_start = i
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                dd_start = current_start
                dd_end = i
        
        return max_dd, dd_start, dd_end
    
    def get_risk_metrics(self, returns: np.ndarray, equity_curve: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # VaR calculations
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # Drawdown
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        current_dd = (equity_curve.max() - equity_curve[-1]) / equity_curve.max()
        
        # Sharpe ratio
        excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
        sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        # Beta and correlation
        if self.market_returns is not None and len(self.market_returns) == len(returns):
            cov = np.cov(returns, self.market_returns)[0, 1]
            market_var = np.var(self.market_returns)
            beta = cov / (market_var + 1e-10)
            correlation = np.corrcoef(returns, self.market_returns)[0, 1]
        else:
            beta = 1.0
            correlation = 0.0
        
        volatility = np.std(returns) * np.sqrt(252)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta=beta,
            correlation_to_market=correlation,
            volatility_annual=volatility
        )
    
    def check_risk_limits(self, returns: np.ndarray, equity_curve: np.ndarray) -> list[str]:
        """Check all risk limits and return violations"""
        violations = []
        
        # Check VaR limit
        daily_var = self.calculate_var(returns[-20:], 0.95) / self.portfolio_value
        if abs(daily_var) > self.limits.daily_var_limit:
            violations.append(f"Daily VaR ({abs(daily_var):.1%}) exceeds limit ({self.limits.daily_var_limit:.1%})")
        
        # Check drawdown limit
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        if max_dd > self.limits.max_drawdown_limit:
            violations.append(f"Max drawdown ({max_dd:.1%}) exceeds limit ({self.limits.max_drawdown_limit:.1%})")
        
        # Check position limits
        for symbol, pos in self.positions.items():
            if pos["weight"] > self.limits.max_single_position_pct:
                violations.append(f"Position {symbol} ({pos['weight']:.1%}) exceeds limit")
        
        # Check sector limits
        for sector, exposure in self.sector_exposures.items():
            weight = exposure / self.portfolio_value
            if weight > self.limits.max_sector_exposure_pct:
                violations.append(f"Sector {sector} ({weight:.1%}) exceeds limit")
        
        return violations


class StressTestEngine:
    """
    Stress testing and scenario analysis.
    """
    
    # Historical crisis scenarios
    SCENARIOS = {
        "2008_financial_crisis": {
            "market_shock": -0.55,
            "volatility_multiplier": 4.0,
            "correlation_increase": 0.3,
            "duration_days": 365
        },
        "2020_covid_crash": {
            "market_shock": -0.34,
            "volatility_multiplier": 5.0,
            "correlation_increase": 0.4,
            "duration_days": 30
        },
        "2000_dotcom_bust": {
            "market_shock": -0.49,
            "volatility_multiplier": 2.5,
            "correlation_increase": 0.2,
            "duration_days": 730
        },
        "flash_crash": {
            "market_shock": -0.10,
            "volatility_multiplier": 10.0,
            "correlation_increase": 0.5,
            "duration_days": 1
        }
    }
    
    def __init__(self, portfolio_value: float, positions: dict[str, dict]):
        self.portfolio_value = portfolio_value
        self.positions = positions
    
    def run_scenario(self, scenario_name: str) -> dict:
        """Run a historical stress scenario"""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.SCENARIOS[scenario_name]
        
        # Calculate impact
        total_loss = 0
        position_impacts = {}
        
        for symbol, pos in self.positions.items():
            beta = pos.get("beta", 1.0)
            position_shock = scenario["market_shock"] * beta
            loss = pos["value"] * position_shock
            total_loss += loss
            position_impacts[symbol] = {
                "loss": loss,
                "loss_pct": position_shock,
                "new_value": pos["value"] + loss
            }
        
        return {
            "scenario": scenario_name,
            "portfolio_loss": total_loss,
            "portfolio_loss_pct": total_loss / self.portfolio_value,
            "new_portfolio_value": self.portfolio_value + total_loss,
            "position_impacts": position_impacts,
            "volatility_impact": scenario["volatility_multiplier"],
            "recovery_days_estimate": scenario["duration_days"]
        }
    
    def run_all_scenarios(self) -> dict:
        """Run all stress scenarios"""
        results = {}
        for scenario_name in self.SCENARIOS:
            results[scenario_name] = self.run_scenario(scenario_name)
        
        # Find worst case
        worst_scenario = min(results.items(), key=lambda x: x[1]["portfolio_loss"])
        
        return {
            "scenarios": results,
            "worst_case": worst_scenario[0],
            "worst_case_loss": worst_scenario[1]["portfolio_loss"],
            "worst_case_loss_pct": worst_scenario[1]["portfolio_loss_pct"]
        }
    
    def monte_carlo_var(
        self,
        returns: np.ndarray,
        num_simulations: int = 10000,
        time_horizon: int = 10,
        confidence: float = 0.95
    ) -> dict:
        """Monte Carlo VaR simulation"""
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        simulated_returns = np.random.normal(
            mu * time_horizon,
            sigma * np.sqrt(time_horizon),
            num_simulations
        )
        
        simulated_values = self.portfolio_value * (1 + simulated_returns)
        
        var = np.percentile(simulated_values, (1 - confidence) * 100) - self.portfolio_value
        cvar = simulated_values[simulated_values <= (self.portfolio_value + var)].mean() - self.portfolio_value
        
        return {
            "var": var,
            "var_pct": var / self.portfolio_value,
            "cvar": cvar,
            "cvar_pct": cvar / self.portfolio_value,
            "time_horizon_days": time_horizon,
            "confidence": confidence,
            "num_simulations": num_simulations,
            "percentile_5": np.percentile(simulated_values, 5),
            "percentile_95": np.percentile(simulated_values, 95),
            "expected_value": np.mean(simulated_values)
        }

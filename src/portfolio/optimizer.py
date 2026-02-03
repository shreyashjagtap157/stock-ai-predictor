"""
Portfolio Optimization Module for Stock AI Predictor
Implements Modern Portfolio Theory (MPT) with Sharpe ratio maximization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MIN_CVAR = "min_cvar"  # Conditional Value at Risk


@dataclass
class Asset:
    """Represents an asset in the portfolio"""
    symbol: str
    name: str = ""
    expected_return: float = 0.0
    volatility: float = 0.0
    weight: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR
    calmar_ratio: float = 0.0


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: Dict[str, float]
    metrics: PortfolioMetrics
    efficient_frontier: Optional[pd.DataFrame] = None
    optimization_success: bool = True
    message: str = ""


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer with multiple objectives.
    
    Supports:
    - Sharpe ratio maximization
    - Minimum volatility
    - Risk parity
    - CVaR optimization
    - Efficient frontier generation
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        allow_short: bool = False
    ):
        self.risk_free_rate = risk_free_rate
        self.default_min_weight = min_weight if not allow_short else -1.0
        self.default_max_weight = max_weight
        self.allow_short = allow_short
        
        self._returns: Optional[pd.DataFrame] = None
        self._cov_matrix: Optional[np.ndarray] = None
        self._mean_returns: Optional[np.ndarray] = None
        self._symbols: List[str] = []
    
    def fit(self, returns: pd.DataFrame):
        """
        Fit the optimizer with historical returns data.
        
        Args:
            returns: DataFrame of daily returns with symbols as columns
        """
        self._returns = returns
        self._symbols = list(returns.columns)
        self._mean_returns = returns.mean().values * 252  # Annualized
        self._cov_matrix = returns.cov().values * 252  # Annualized
        
        logger.info(f"Fitted optimizer with {len(self._symbols)} assets, "
                   f"{len(returns)} observations")
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return np.dot(weights, self._mean_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self._cov_matrix, weights)))
    
    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio (negative for minimization)"""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol
    
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe for minimization"""
        return -self._portfolio_sharpe(weights)
    
    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """Risk parity: equal risk contribution from each asset"""
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 1e10
        
        # Marginal risk contribution
        marginal_contrib = np.dot(self._cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / vol
        
        # Target: equal contribution
        target = vol / len(weights)
        return np.sum((risk_contrib - target) ** 2)
    
    def _cvar_objective(self, weights: np.ndarray, alpha: float = 0.05) -> float:
        """Conditional Value at Risk (CVaR / Expected Shortfall)"""
        if self._returns is None:
            return 1e10
        
        portfolio_returns = self._returns.values @ weights
        var = np.percentile(portfolio_returns, alpha * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return -cvar  # Minimize negative CVaR = maximize CVaR (less negative = better)
    
    def _get_constraints(self) -> List[Dict]:
        """Get optimization constraints"""
        return [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    def _get_bounds(self, assets: Optional[List[Asset]] = None) -> List[Tuple[float, float]]:
        """Get weight bounds for each asset"""
        n = len(self._symbols)
        
        if assets:
            return [(a.min_weight, a.max_weight) for a in assets]
        
        return [(self.default_min_weight, self.default_max_weight)] * n
    
    def optimize(
        self,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        assets: Optional[List[Asset]] = None,
        method: str = "SLSQP"
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            objective: Optimization objective
            assets: Optional list of Asset objects with constraints
            method: Optimization method (SLSQP, trust-constr, etc.)
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        if self._returns is None:
            raise ValueError("Must call fit() first with returns data")
        
        n = len(self._symbols)
        initial_weights = np.array([1.0 / n] * n)
        bounds = self._get_bounds(assets)
        constraints = self._get_constraints()
        
        # Select objective function
        if objective == OptimizationObjective.MAX_SHARPE:
            obj_func = self._negative_sharpe
        elif objective == OptimizationObjective.MIN_VOLATILITY:
            obj_func = self._portfolio_volatility
        elif objective == OptimizationObjective.MAX_RETURN:
            obj_func = lambda w: -self._portfolio_return(w)
        elif objective == OptimizationObjective.RISK_PARITY:
            obj_func = self._risk_parity_objective
        elif objective == OptimizationObjective.MIN_CVAR:
            obj_func = self._cvar_objective
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Run optimization
        result = minimize(
            obj_func,
            initial_weights,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization may not have converged: {result.message}")
        
        weights = result.x
        
        # Calculate metrics
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights={self._symbols[i]: float(weights[i]) for i in range(n)},
            metrics=metrics,
            optimization_success=result.success,
            message=result.message
        )
    
    def _calculate_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        exp_return = self._portfolio_return(weights)
        volatility = self._portfolio_volatility(weights)
        sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate additional metrics if returns data available
        sortino = 0.0
        max_drawdown = 0.0
        var_95 = 0.0
        cvar_95 = 0.0
        
        if self._returns is not None:
            portfolio_returns = self._returns.values @ weights
            
            # Sortino (downside deviation)
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else volatility
            sortino = (exp_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdowns))
            
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else var_95
        
        calmar = exp_return / max_drawdown if max_drawdown > 0 else 0
        
        return PortfolioMetrics(
            expected_return=float(exp_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            calmar_ratio=float(calmar)
        )
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        return_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Returns DataFrame with columns: return, volatility, sharpe, weights
        """
        if self._returns is None:
            raise ValueError("Must call fit() first")
        
        # Determine return range
        if return_range is None:
            min_ret = self._mean_returns.min()
            max_ret = self._mean_returns.max()
        else:
            min_ret, max_ret = return_range
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        results = []
        for target_ret in target_returns:
            try:
                result = self._optimize_for_target_return(target_ret)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Failed to optimize for return {target_ret}: {e}")
        
        df = pd.DataFrame(results)
        
        # Remove dominated points (keeping efficient frontier only)
        df = df.sort_values('volatility')
        efficient = [0]
        max_return = df.iloc[0]['return']
        
        for i in range(1, len(df)):
            if df.iloc[i]['return'] > max_return:
                efficient.append(i)
                max_return = df.iloc[i]['return']
        
        return df.iloc[efficient].reset_index(drop=True)
    
    def _optimize_for_target_return(self, target_return: float) -> Optional[Dict]:
        """Optimize portfolio for a target return level"""
        n = len(self._symbols)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target_return}
        ]
        
        result = minimize(
            self._portfolio_volatility,
            np.array([1.0 / n] * n),
            method='SLSQP',
            bounds=self._get_bounds(),
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            vol = self._portfolio_volatility(weights)
            ret = self._portfolio_return(weights)
            sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
            
            return {
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe,
                'weights': {self._symbols[i]: weights[i] for i in range(n)}
            }
        return None
    
    def black_litterman(
        self,
        views: Dict[str, float],
        view_confidence: Dict[str, float],
        tau: float = 0.05
    ) -> OptimizationResult:
        """
        Black-Litterman model incorporating investor views.
        
        Args:
            views: Dict of symbol -> expected return view
            view_confidence: Dict of symbol -> confidence (0-1)
            tau: Uncertainty scaling factor
        
        Returns:
            OptimizationResult with BL-adjusted weights
        """
        if self._returns is None:
            raise ValueError("Must call fit() first")
        
        n = len(self._symbols)
        
        # Market equilibrium returns (reverse optimization from market cap weights)
        # Using equal weights as proxy for market weights
        market_weights = np.array([1.0 / n] * n)
        lambda_risk_aversion = (self._mean_returns.mean() - self.risk_free_rate) / (market_weights @ self._cov_matrix @ market_weights)
        pi = lambda_risk_aversion * self._cov_matrix @ market_weights
        
        # Build view matrices
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        omega_diag = []
        
        for i, (symbol, view_return) in enumerate(views.items()):
            if symbol in self._symbols:
                idx = self._symbols.index(symbol)
                P[i, idx] = 1.0
                Q[i] = view_return
                conf = view_confidence.get(symbol, 0.5)
                omega_diag.append((1 - conf) * tau * self._cov_matrix[idx, idx])
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau_sigma = tau * self._cov_matrix
        M1 = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
        M2 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
        
        bl_returns = M1 @ M2
        
        # Optimize with BL returns
        original_mean = self._mean_returns.copy()
        self._mean_returns = bl_returns
        
        result = self.optimize(OptimizationObjective.MAX_SHARPE)
        
        self._mean_returns = original_mean
        
        return result
    
    def rebalance_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        transaction_cost: float = 0.001
    ) -> float:
        """Calculate rebalancing cost"""
        total_turnover = 0.0
        for symbol in self._symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            total_turnover += abs(target - current)
        
        return total_turnover * transaction_cost


def calculate_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame of prices with symbols as columns
        method: 'log' for log returns, 'simple' for simple returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()

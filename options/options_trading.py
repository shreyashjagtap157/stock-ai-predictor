"""
Options Trading Module for Stock-AI

Implements Black-Scholes model, Greeks calculations, and options strategies.

Features:
- Black-Scholes option pricing
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Strategy builder (spreads, straddles, etc.)
- Implied volatility calculation
- Risk metrics and position management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type"""
    CALL = "call"
    PUT = "put"


class StrategyType(Enum):
    """Options strategies"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    IRON_CONDOR = "iron_condor"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"


@dataclass
class OptionContract:
    """Option contract specification"""
    symbol: str
    option_type: OptionType
    strike_price: float
    expiry_date: datetime
    current_price: float  # Current underlying price
    volatility: float  # Implied volatility (%)
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    quantity: int = 1
    position_type: str = "long"  # long or short


@dataclass
class GreekValues:
    """Greeks for an option"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    price: float


@dataclass
class StrategyPosition:
    """Position in options strategy"""
    strategy_type: StrategyType
    contracts: List[OptionContract]
    entry_price: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]


class BlackScholesCalculator:
    """
    Implements Black-Scholes option pricing model.
    """
    
    @staticmethod
    def d1(
        S: float,  # Current price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        q: float = 0.0  # Dividend yield
    ) -> float:
        """Calculate d1 parameter"""
        return (
            np.log(S / K) +
            (r - q + 0.5 * sigma ** 2) * T
        ) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter"""
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """Calculate call option price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesCalculator.d2(d1, sigma, T)
        
        call = (
            S * np.exp(-q * T) * stats.norm.cdf(d1) -
            K * np.exp(-r * T) * stats.norm.cdf(d2)
        )
        
        return max(call, 0)
    
    @staticmethod
    def put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """Calculate put option price"""
        if T <= 0:
            return max(K - S, 0)
        
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesCalculator.d2(d1, sigma, T)
        
        put = (
            K * np.exp(-r * T) * stats.norm.cdf(-d2) -
            S * np.exp(-q * T) * stats.norm.cdf(-d1)
        )
        
        return max(put, 0)


class GreeksCalculator:
    """
    Calculates option Greeks (Delta, Gamma, Vega, Theta, Rho).
    """
    
    @staticmethod
    def delta(
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Delta: rate of change of option price w.r.t. underlying price
        Range: [-1, 1] for puts, [0, 1] for calls
        """
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        
        if option_type == OptionType.CALL:
            return np.exp(-q * T) * stats.norm.cdf(d1)
        else:  # PUT
            return np.exp(-q * T) * (stats.norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Gamma: rate of change of delta w.r.t. underlying price
        Measures convexity/curvature of option price
        """
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        
        gamma = (
            np.exp(-q * T) * stats.norm.pdf(d1) /
            (S * sigma * np.sqrt(T))
        )
        
        return gamma
    
    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Vega: sensitivity to changes in volatility (per 1% change)
        Positive for both calls and puts
        """
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        
        vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100.0
        
        return vega
    
    @staticmethod
    def theta(
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Theta: time decay (per day)
        Usually negative for long options (decay works against holder)
        """
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesCalculator.d2(d1, sigma, T)
        
        if option_type == OptionType.CALL:
            theta = (
                -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * stats.norm.cdf(d2) +
                q * S * np.exp(-q * T) * stats.norm.cdf(d1)
            ) / 365.0
        else:  # PUT
            theta = (
                -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * stats.norm.cdf(-d2) -
                q * S * np.exp(-q * T) * stats.norm.cdf(-d1)
            ) / 365.0
        
        return theta
    
    @staticmethod
    def rho(
        option_type: OptionType,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Rho: sensitivity to interest rate changes (per 1% change)
        """
        d2 = BlackScholesCalculator.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
        
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100.0
        else:  # PUT
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100.0
        
        return rho


class OptionsPricer:
    """
    Prices options and calculates Greeks.
    """
    
    def __init__(self):
        self.calculator = BlackScholesCalculator()
        self.greeks_calc = GreeksCalculator()
    
    def price_option(self, contract: OptionContract) -> GreekValues:
        """
        Price option and calculate Greeks.
        """
        # Calculate time to expiry in years
        T = (contract.expiry_date - datetime.now()).days / 365.0
        
        if T <= 0:
            # Option expired
            if contract.option_type == OptionType.CALL:
                price = max(contract.current_price - contract.strike_price, 0)
            else:
                price = max(contract.strike_price - contract.current_price, 0)
            
            return GreekValues(0, 0, 0, 0, 0, price)
        
        # Convert volatility from percentage to decimal
        sigma = contract.volatility / 100.0
        
        # Calculate price
        if contract.option_type == OptionType.CALL:
            price = self.calculator.call_price(
                contract.current_price,
                contract.strike_price,
                T,
                contract.risk_free_rate,
                sigma,
                contract.dividend_yield
            )
        else:
            price = self.calculator.put_price(
                contract.current_price,
                contract.strike_price,
                T,
                contract.risk_free_rate,
                sigma,
                contract.dividend_yield
            )
        
        # Calculate Greeks
        delta = self.greeks_calc.delta(
            contract.option_type,
            contract.current_price,
            contract.strike_price,
            T,
            contract.risk_free_rate,
            sigma,
            contract.dividend_yield
        )
        
        gamma = self.greeks_calc.gamma(
            contract.current_price,
            contract.strike_price,
            T,
            contract.risk_free_rate,
            sigma,
            contract.dividend_yield
        )
        
        vega = self.greeks_calc.vega(
            contract.current_price,
            contract.strike_price,
            T,
            contract.risk_free_rate,
            sigma,
            contract.dividend_yield
        )
        
        theta = self.greeks_calc.theta(
            contract.option_type,
            contract.current_price,
            contract.strike_price,
            T,
            contract.risk_free_rate,
            sigma,
            contract.dividend_yield
        )
        
        rho = self.greeks_calc.rho(
            contract.option_type,
            contract.current_price,
            contract.strike_price,
            T,
            contract.risk_free_rate,
            sigma,
            contract.dividend_yield
        )
        
        return GreekValues(delta, gamma, vega, theta, rho, price)


class OptionsStrategyBuilder:
    """
    Builds and analyzes options strategies.
    """
    
    def __init__(self, pricer: OptionsPricer):
        self.pricer = pricer
    
    def build_call_spread(
        self,
        long_call: OptionContract,
        short_call: OptionContract
    ) -> StrategyPosition:
        """Bull call spread: long call at lower strike, short call at higher strike"""
        long_price = self.pricer.price_option(long_call).price
        short_price = self.pricer.price_option(short_call).price
        
        entry_cost = (long_price - short_price) * 100  # 100 contracts per option
        max_profit = (short_call.strike_price - long_call.strike_price) * 100 - entry_cost
        max_loss = entry_cost
        
        breakeven = long_call.strike_price + entry_cost / 100
        
        return StrategyPosition(
            strategy_type=StrategyType.CALL_SPREAD,
            contracts=[long_call, short_call],
            entry_price=entry_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven]
        )
    
    def build_straddle(
        self,
        call: OptionContract,
        put: OptionContract
    ) -> StrategyPosition:
        """Long straddle: long call and put at same strike (volatility play)"""
        call_price = self.pricer.price_option(call).price
        put_price = self.pricer.price_option(put).price
        
        entry_cost = (call_price + put_price) * 100
        strike = call.strike_price
        
        max_loss = entry_cost
        breakeven_up = strike + entry_cost / 100
        breakeven_down = strike - entry_cost / 100
        
        return StrategyPosition(
            strategy_type=StrategyType.STRADDLE,
            contracts=[call, put],
            entry_price=entry_cost,
            max_profit=float('inf'),  # Theoretically unlimited
            max_loss=max_loss,
            breakeven_points=[breakeven_down, breakeven_up]
        )
    
    def build_iron_condor(
        self,
        long_put: OptionContract,
        short_put: OptionContract,
        short_call: OptionContract,
        long_call: OptionContract
    ) -> StrategyPosition:
        """Iron condor: sell call spread and put spread"""
        long_put_price = self.pricer.price_option(long_put).price
        short_put_price = self.pricer.price_option(short_put).price
        short_call_price = self.pricer.price_option(short_call).price
        long_call_price = self.pricer.price_option(long_call).price
        
        # Credit received
        credit = (short_put_price - long_put_price + short_call_price - long_call_price) * 100
        
        # Width of spreads
        put_width = short_put.strike_price - long_put.strike_price
        call_width = long_call.strike_price - short_call.strike_price
        
        max_profit = credit
        max_loss = min(put_width, call_width) * 100 - credit
        
        return StrategyPosition(
            strategy_type=StrategyType.IRON_CONDOR,
            contracts=[long_put, short_put, short_call, long_call],
            entry_price=-credit,  # Negative because credit received
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[
                long_put.strike_price + credit / 100,
                long_call.strike_price - credit / 100
            ]
        )


class OptionsTrader:
    """
    Main options trading interface.
    """
    
    def __init__(self):
        self.pricer = OptionsPricer()
        self.builder = OptionsStrategyBuilder(self.pricer)
        self.positions: List[StrategyPosition] = []
        self.trade_history: List[Dict[str, Any]] = []
    
    def analyze_position(
        self,
        contracts: List[OptionContract]
    ) -> Dict[str, Any]:
        """
        Analyze a position's Greeks and risk metrics.
        """
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
            "price": 0.0
        }
        
        for contract in contracts:
            greeks = self.pricer.price_option(contract)
            multiplier = contract.quantity * (1 if contract.position_type == "long" else -1)
            
            portfolio_greeks["delta"] += greeks.delta * multiplier * 100
            portfolio_greeks["gamma"] += greeks.gamma * multiplier * 100
            portfolio_greeks["vega"] += greeks.vega * multiplier
            portfolio_greeks["theta"] += greeks.theta * multiplier * 100
            portfolio_greeks["rho"] += greeks.rho * multiplier
            portfolio_greeks["price"] += greeks.price * multiplier * 100
        
        return portfolio_greeks
    
    def simulate_pnl(
        self,
        contracts: List[OptionContract],
        underlying_moves: List[float]
    ) -> List[float]:
        """
        Simulate P&L across range of underlying price movements.
        """
        pnl_curve = []
        original_price = contracts[0].current_price
        
        for move in underlying_moves:
            pnl = 0.0
            
            for contract in contracts:
                # Update price
                contract.current_price = original_price + move
                greeks = self.pricer.price_option(contract)
                
                multiplier = contract.quantity * (1 if contract.position_type == "long" else -1)
                pnl += greeks.price * multiplier * 100
            
            pnl_curve.append(pnl)
            
            # Restore original price
            for contract in contracts:
                contract.current_price = original_price
        
        return pnl_curve

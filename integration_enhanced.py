"""
Integration of ensemble prediction, options trading, microstructure analysis,
and backtesting into Stock-AI's main trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import numpy as np

from ensemble.meta_learner import EnsemblePredictor, PredictionInput
from options.options_trading import OptionsTrader, OptionsPricer, OptionsStrategyBuilder
from microstructure.market_microstructure import MicrostructureAnalyzer, OrderBook
from backtesting.backtest_engine import BacktestEngine, Portfolio, Position, Order

logger = logging.getLogger(__name__)


class EnhancedTradingSystem:
    """
    Enhanced trading system with ensemble predictions, options strategies,
    microstructure analysis, and backtesting capabilities.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        symbols: Optional[List[str]] = None,
        enable_ensemble: bool = True,
        enable_options: bool = True,
        enable_microstructure: bool = True,
        enable_backtesting: bool = True,
    ):
        self.initial_capital = initial_capital
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        # Initialize ensemble predictor
        self.ensemble_predictor: Optional[EnsemblePredictor] = None
        if enable_ensemble:
            self.ensemble_predictor = EnsemblePredictor(
                models=["xgboost", "lightgbm", "catboost"],
                use_regime_detection=True
            )
        
        # Initialize options trading
        self.options_trader: Optional[OptionsTrader] = None
        self.options_pricer: Optional[OptionsPricer] = None
        self.strategy_builder: Optional[OptionsStrategyBuilder] = None
        if enable_options:
            self.options_pricer = OptionsPricer()
            self.strategy_builder = OptionsStrategyBuilder()
            self.options_trader = OptionsTrader(
                capital=initial_capital * 0.1,  # 10% for options
                risk_per_trade=0.02
            )
        
        # Initialize microstructure analyzer
        self.microstructure: Optional[MicrostructureAnalyzer] = None
        if enable_microstructure:
            self.microstructure = MicrostructureAnalyzer()
            # Initialize order books for each symbol
            self._order_books: Dict[str, OrderBook] = {
                symbol: OrderBook(symbol=symbol) for symbol in self.symbols
            }
        
        # Initialize backtest engine
        self.backtest_engine: Optional[BacktestEngine] = None
        if enable_backtesting:
            self.backtest_engine = BacktestEngine(
                portfolio=Portfolio(
                    initial_capital=initial_capital,
                    commission_pct=0.001,
                    slippage_pct=0.0005
                )
            )
        
        self.live_portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
    
    async def generate_trading_signals(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid_volume: float,
        ask_volume: float,
        bid_price: float,
        ask_price: float,
        historical_prices: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate trading signals using ensemble prediction and microstructure analysis.
        """
        signals = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'ensemble_prediction': None,
            'microstructure_analysis': None,
            'options_opportunity': None,
            'backtest_recommendation': None,
            'action': 'HOLD'
        }
        
        # Ensemble prediction
        if self.ensemble_predictor and historical_prices is not None:
            try:
                prediction_input = PredictionInput(
                    symbol=symbol,
                    features=historical_prices[-20:],  # Last 20 price points
                    timestamp=datetime.utcnow()
                )
                ensemble_pred = self.ensemble_predictor.predict(prediction_input)
                signals['ensemble_prediction'] = {
                    'direction': ensemble_pred.direction.value,
                    'confidence': ensemble_pred.confidence,
                    'regime': ensemble_pred.regime.value,
                    'probability': ensemble_pred.probability
                }
                logger.debug(f"Ensemble prediction for {symbol}: {ensemble_pred.direction.value}")
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}")
        
        # Microstructure analysis
        if self.microstructure and symbol in self._order_books:
            try:
                order_book = self._order_books[symbol]
                # Add order book data
                for _ in range(int(bid_volume)):
                    order_book.add_bid(bid_price)
                for _ in range(int(ask_volume)):
                    order_book.add_ask(ask_price)
                
                analysis = self.microstructure.analyze_order_book(order_book)
                signals['microstructure_analysis'] = {
                    'spread': analysis['spread'],
                    'spread_pct': analysis['spread_pct'],
                    'volume_imbalance': analysis['volume_imbalance'],
                    'liquidity_score': analysis['liquidity_score']
                }
                logger.debug(f"Microstructure: spread={analysis['spread']:.4f}, "
                           f"imbalance={analysis['volume_imbalance']:.2f}")
            except Exception as e:
                logger.warning(f"Microstructure analysis failed: {e}")
        
        # Options opportunities
        if self.options_trader and self.options_pricer:
            try:
                # Check for profitable options strategies
                call_price = self.options_pricer.price_option(
                    S=price, K=price, T=30/365, r=0.05, sigma=0.2, option_type='call'
                )
                put_price = self.options_pricer.price_option(
                    S=price, K=price, T=30/365, r=0.05, sigma=0.2, option_type='put'
                )
                
                signals['options_opportunity'] = {
                    'call_price': float(call_price),
                    'put_price': float(put_price),
                    'atm_spread': float(call_price + put_price)
                }
                logger.debug(f"Options: call={call_price:.2f}, put={put_price:.2f}")
            except Exception as e:
                logger.warning(f"Options analysis failed: {e}")
        
        # Determine trading action
        if signals['ensemble_prediction']:
            pred = signals['ensemble_prediction']
            if pred['direction'] == 'UP' and pred['confidence'] > 0.65:
                if signals['microstructure_analysis'] and \
                   signals['microstructure_analysis']['liquidity_score'] > 0.7:
                    signals['action'] = 'BUY'
            elif pred['direction'] == 'DOWN' and pred['confidence'] > 0.65:
                if signals['microstructure_analysis'] and \
                   signals['microstructure_analysis']['liquidity_score'] > 0.7:
                    signals['action'] = 'SELL'
        
        return signals
    
    async def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str = 'market'
    ) -> bool:
        """Execute a trade and record it in backtesting engine."""
        try:
            order = Order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                timestamp=datetime.utcnow()
            )
            
            # Add to backtesting engine for historical analysis
            if self.backtest_engine:
                if action == 'BUY':
                    self.backtest_engine.execute_buy_order(order)
                elif action == 'SELL':
                    self.backtest_engine.execute_sell_order(order)
            
            # Execute in live portfolio
            if action == 'BUY':
                self.live_portfolio.buy(symbol, quantity, price)
            elif action == 'SELL':
                self.live_portfolio.sell(symbol, quantity, price)
            
            logger.info(f"Trade executed: {action} {quantity} {symbol} @ {price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary with backtest metrics."""
        summary = {
            'live_portfolio': {
                'total_value': self.live_portfolio.total_value,
                'cash': self.live_portfolio.cash,
                'positions': len(self.live_portfolio.positions),
                'pnl': self.live_portfolio.get_pnl(),
                'return_pct': self.live_portfolio.get_return_percentage()
            }
        }
        
        if self.backtest_engine:
            backtest_stats = self.backtest_engine.get_statistics()
            summary['backtest_metrics'] = backtest_stats
        
        return summary
    
    async def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        price_data: np.ndarray
    ) -> Dict:
        """
        Run backtest on historical data.
        """
        if not self.backtest_engine:
            return {'error': 'Backtesting not enabled'}
        
        results = {
            'symbol': symbol,
            'period': f"{start_date.date()} to {end_date.date()}",
            'trades': [],
            'metrics': {}
        }
        
        try:
            # Simulate trading on price data
            for i in range(1, len(price_data)):
                price = price_data[i]
                prev_price = price_data[i - 1]
                
                # Generate signal
                signal_data = await self.generate_trading_signals(
                    symbol=symbol,
                    price=price,
                    volume=1000,
                    bid_volume=500,
                    ask_volume=500,
                    bid_price=price * 0.9999,
                    ask_price=price * 1.0001,
                    historical_prices=price_data[:i]
                )
                
                # Execute if signal
                if signal_data['action'] in ['BUY', 'SELL']:
                    order = Order(
                        symbol=symbol,
                        quantity=10,
                        price=price,
                        order_type='market',
                        timestamp=start_date + timedelta(days=i)
                    )
                    
                    if signal_data['action'] == 'BUY':
                        self.backtest_engine.execute_buy_order(order)
                    else:
                        self.backtest_engine.execute_sell_order(order)
                    
                    results['trades'].append({
                        'timestamp': order.timestamp.isoformat(),
                        'action': signal_data['action'],
                        'price': price
                    })
            
            metrics = self.backtest_engine.get_statistics()
            results['metrics'] = metrics
            logger.info(f"Backtest completed: {len(results['trades'])} trades, "
                       f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            results['error'] = str(e)
        
        return results


async def main():
    """Main trading system initialization."""
    trading_system = EnhancedTradingSystem(
        initial_capital=100000.0,
        symbols=["AAPL", "MSFT", "GOOGL"],
        enable_ensemble=True,
        enable_options=True,
        enable_microstructure=True,
        enable_backtesting=True
    )
    
    logger.info("Enhanced trading system initialized")
    logger.info(f"Initial capital: ${trading_system.initial_capital:,.2f}")
    
    # Example: Generate signals for a stock
    signals = await trading_system.generate_trading_signals(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        bid_volume=500000,
        ask_volume=500000,
        bid_price=149.99,
        ask_price=150.01,
        historical_prices=np.array([140.0, 142.0, 145.0, 148.0, 150.0])
    )
    logger.info(f"Generated signals: {signals}")
    
    # Example: Get portfolio summary
    summary = await trading_system.get_portfolio_summary()
    logger.info(f"Portfolio summary: {summary}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())

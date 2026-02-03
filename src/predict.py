"""
Simple inference pipeline. Loads model and runs prediction on latest window.
Enhanced with ensemble predictions, options analysis, and microstructure awareness.
"""
import yaml
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import logging
from src.features.indicators import add_features
from src.models.lstm_model import create_model, load_model

# Import enhancements
try:
    from ensemble.meta_learner import EnsemblePredictor, PredictionInput
    from options.options_trading import OptionsPricer, OptionsStrategyBuilder
    from microstructure.market_microstructure import MicrostructureAnalyzer, OrderBook
    from backtesting.backtest_engine import BacktestEngine, Portfolio
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

CONFIG_PATH = Path(__file__).parents[2] / 'config.yaml'
logger = logging.getLogger(__name__)

# Global enhancement instances
ensemble_predictor = None
options_pricer = None
strategy_builder = None
microstructure_analyzer = None
backtest_engine = None

def load_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}

def initialize_enhancements():
    """Initialize all trading enhancements."""
    global ensemble_predictor, options_pricer, strategy_builder, microstructure_analyzer, backtest_engine
    
    if not ENHANCEMENTS_AVAILABLE:
        logger.warning("Enhancements not available, running in basic mode")
        return
    
    try:
        config = load_config()
        
        # Initialize ensemble predictor
        if config.get("ensemble_prediction", {}).get("enabled", False):
            ensemble_predictor = EnsemblePredictor(
                models=config.get("ensemble_prediction", {}).get("models", ["xgboost", "lightgbm"]),
                use_regime_detection=config.get("ensemble_prediction", {}).get("use_regime_detection", True)
            )
            logger.info("Ensemble predictor initialized")
        
        # Initialize options pricer and strategy builder
        if config.get("options_trading", {}).get("enabled", False):
            options_pricer = OptionsPricer()
            strategy_builder = OptionsStrategyBuilder()
            logger.info("Options trading initialized")
        
        # Initialize microstructure analyzer
        if config.get("microstructure_analysis", {}).get("enabled", False):
            microstructure_analyzer = MicrostructureAnalyzer()
            logger.info("Microstructure analyzer initialized")
        
        # Initialize backtest engine
        if config.get("backtesting", {}).get("enabled", False):
            initial_capital = config.get("backtesting", {}).get("initial_capital", 100000.0)
            backtest_engine = BacktestEngine(
                portfolio=Portfolio(
                    initial_capital=initial_capital,
                    commission_pct=config.get("backtesting", {}).get("commission_pct", 0.001),
                    slippage_pct=config.get("backtesting", {}).get("slippage_pct", 0.0005)
                )
            )
            logger.info("Backtest engine initialized")
    
    except Exception as e:
        logger.error(f"Error initializing enhancements: {e}")



def predict_from_df(model, df, seq_len=32, feature_cols=None, symbol="UNKNOWN", price=None):
    """
    Enhanced prediction with ensemble, options, and microstructure analysis.
    """
    df = add_features(df)
    if feature_cols is None:
        feature_cols = ['close','sma_10','sma_50','rsi_14','macd_hist','ret_1','ret_5']
    arr = df[feature_cols].values.astype(np.float32)
    if len(arr) < seq_len:
        raise ValueError('Not enough data')
    window = arr[-seq_len:]
    x = np.expand_dims(window, 0)  # batch=1
    x = torch.from_numpy(x)
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    base_prediction = out.numpy().ravel()[0]
    
    # Get current price if available
    if price is None and len(df) > 0:
        price = df['close'].iloc[-1]
    
    result = {
        'base_prediction': base_prediction,
        'ensemble_prediction': None,
        'options_analysis': None,
        'microstructure_analysis': None,
        'final_signal': 'HOLD'
    }
    
    # Enhance with ensemble prediction
    if ensemble_predictor is not None:
        try:
            price_array = df['close'].values.astype(np.float32)
            prediction_input = PredictionInput(
                symbol=symbol,
                features=price_array[-20:] if len(price_array) >= 20 else price_array,
                timestamp=datetime.utcnow()
            )
            ensemble_pred = ensemble_predictor.predict(prediction_input)
            result['ensemble_prediction'] = {
                'direction': ensemble_pred.direction.value,
                'confidence': ensemble_pred.confidence,
                'regime': ensemble_pred.regime.value
            }
            logger.debug(f"Ensemble for {symbol}: {ensemble_pred.direction.value} ({ensemble_pred.confidence:.2%})")
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
    
    # Analyze options opportunities
    if options_pricer is not None and price is not None:
        try:
            call_price = options_pricer.price_option(
                S=price, K=price, T=30/365, r=0.05, sigma=0.2, option_type='call'
            )
            put_price = options_pricer.price_option(
                S=price, K=price, T=30/365, r=0.05, sigma=0.2, option_type='put'
            )
            result['options_analysis'] = {
                'call_price': float(call_price),
                'put_price': float(put_price)
            }
            logger.debug(f"Options for {symbol}: call={call_price:.2f}, put={put_price:.2f}")
        except Exception as e:
            logger.warning(f"Options analysis failed: {e}")
    
    # Determine trading signal
    if result['ensemble_prediction']:
        if result['ensemble_prediction']['direction'] == 'UP' and result['ensemble_prediction']['confidence'] > 0.65:
            result['final_signal'] = 'BUY'
        elif result['ensemble_prediction']['direction'] == 'DOWN' and result['ensemble_prediction']['confidence'] > 0.65:
            result['final_signal'] = 'SELL'
    
    return result

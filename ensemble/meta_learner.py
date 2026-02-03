"""
Ensemble Meta-Learner for Stock-AI

Combines multiple base learners (RL agents, regime detectors, sentiment models)
with a meta-learner that dynamically weights predictions based on market conditions.

Features:
- XGBoost/LightGBM meta-learner
- Adaptive weighting based on market regime
- Cross-validation stacking
- Online learning and retraining
- Performance tracking per regime
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERT = "mean_revert"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"


@dataclass
class PredictionInput:
    """Input features for ensemble prediction"""
    timestamp: datetime
    price: float
    volume: float
    volatility: float
    
    # Base model predictions
    rl_agent_action: float  # -1 to 1
    regime_prob: Dict[str, float]  # Probability for each regime
    sentiment_score: float  # -1 to 1
    technical_signal: float  # -1 to 1
    
    # Market features
    rsi: float
    macd: float
    bollinger_position: float
    trend_strength: float
    
    # Features for meta-learner
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Output of ensemble prediction"""
    timestamp: datetime
    predicted_direction: float  # -1 to 1
    predicted_return: float  # Expected return %
    confidence: float  # 0 to 1
    base_predictions: Dict[str, float]  # Individual model predictions
    weights: Dict[str, float]  # Weights assigned to each model
    regime: MarketRegime
    recommended_action: str  # BUY, SELL, HOLD


class BaseModel:
    """Interface for base models"""
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Return prediction in range [-1, 1]"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Return model name"""
        raise NotImplementedError


class RLAgentPredictor(BaseModel):
    """Wraps RL agent for ensemble"""
    
    def __init__(self, agent_id: str = "dqn_agent"):
        self.agent_id = agent_id
        self.policy_network = None
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Get action from RL agent"""
        # In practice, would call actual RL agent
        return np.tanh(np.random.randn())
    
    def get_name(self) -> str:
        return f"rl_agent_{self.agent_id}"


class RegimeDetectorPredictor(BaseModel):
    """Wraps regime detector for ensemble"""
    
    def __init__(self):
        self.hmm_model = None
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict based on regime"""
        regime = features.get("current_regime", MarketRegime.RANGING)
        
        # Bias predictions based on regime
        if regime == MarketRegime.TRENDING_UP:
            return 0.7
        elif regime == MarketRegime.TRENDING_DOWN:
            return -0.7
        elif regime == MarketRegime.MEAN_REVERT:
            # Contrarian approach
            return -features.get("recent_return", 0) * 0.5
        else:
            return 0.0
    
    def get_name(self) -> str:
        return "regime_detector"


class SentimentPredictor(BaseModel):
    """Sentiment analysis predictor"""
    
    def __init__(self):
        self.model = None
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict based on sentiment"""
        # Would use actual sentiment model
        return features.get("sentiment_score", 0.0)
    
    def get_name(self) -> str:
        return "sentiment_model"


class TechnicalIndicatorPredictor(BaseModel):
    """Technical analysis predictor"""
    
    def __init__(self):
        self.weights = {}
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict based on technical indicators"""
        signal = 0.0
        
        # RSI signal
        rsi = features.get("rsi", 50)
        if rsi > 70:
            signal -= 0.3  # Overbought
        elif rsi < 30:
            signal += 0.3  # Oversold
        
        # MACD signal
        signal += np.tanh(features.get("macd", 0) / 100) * 0.3
        
        # Bollinger Bands signal
        bb_pos = features.get("bollinger_position", 0.5)
        if bb_pos > 0.8:
            signal -= 0.2
        elif bb_pos < 0.2:
            signal += 0.2
        
        return np.clip(signal, -1, 1)
    
    def get_name(self) -> str:
        return "technical_indicators"


class MetaLearner:
    """
    Meta-learner that combines base model predictions.
    Uses XGBoost-like gradient boosting for optimal weighting.
    """
    
    def __init__(
        self,
        num_base_models: int = 4,
        learning_rate: float = 0.1,
        max_iterations: int = 100
    ):
        self.num_base_models = num_base_models
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Meta-learner parameters
        self.weights: np.ndarray = np.ones(num_base_models) / num_base_models
        self.regime_weights: Dict[MarketRegime, np.ndarray] = {}
        self.training_history: List[Dict[str, Any]] = []
    
    def predict(
        self,
        base_predictions: np.ndarray,
        regime: Optional[MarketRegime] = None
    ) -> Tuple[float, float]:
        """
        Predict using base model predictions.
        
        Returns: (prediction, confidence)
        """
        # Use regime-specific weights if available
        if regime and regime in self.regime_weights:
            weights = self.regime_weights[regime]
        else:
            weights = self.weights
        
        # Weighted combination
        prediction = np.dot(base_predictions, weights)
        
        # Confidence based on agreement among models
        disagreement = np.std(base_predictions)
        confidence = 1.0 / (1.0 + disagreement)
        
        return float(np.clip(prediction, -1, 1)), float(confidence)
    
    def train(
        self,
        X: np.ndarray,  # Base predictions (N, num_models)
        y: np.ndarray,  # Target returns (N,)
        regimes: Optional[List[MarketRegime]] = None,
        validation_split: float = 0.2
    ):
        """
        Train meta-learner on base model predictions.
        
        X shape: (num_samples, num_base_models)
        y shape: (num_samples,)
        """
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Normalize inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Simple gradient descent on mean squared error
        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = X_train @ self.weights
            residuals = predictions - y_train
            
            # Backward pass
            gradient = X_train.T @ residuals / len(X_train)
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Normalize weights to sum to 1
            self.weights = np.abs(self.weights) / (np.sum(np.abs(self.weights)) + 1e-10)
            
            # Validation
            if iteration % 10 == 0:
                val_pred = X_val @ self.weights
                val_mse = mean_squared_error(y_val, val_pred)
                
                self.training_history.append({
                    "iteration": iteration,
                    "train_mse": float(mean_squared_error(y_train, predictions)),
                    "val_mse": float(val_mse)
                })
        
        # Train regime-specific weights
        if regimes:
            unique_regimes = set(regimes)
            for regime in unique_regimes:
                regime_mask = np.array(regimes) == regime
                
                if np.sum(regime_mask) > 0:
                    X_regime = X_train[regime_mask]
                    y_regime = y_train[regime_mask]
                    
                    # Simple linear regression for this regime
                    if len(X_regime) > 0:
                        regime_weights = np.linalg.lstsq(X_regime, y_regime, rcond=None)[0]
                        regime_weights = np.abs(regime_weights) / (np.sum(np.abs(regime_weights)) + 1e-10)
                        self.regime_weights[regime] = regime_weights
        
        logger.info(f"Meta-learner trained on {len(X_train)} samples")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        weights_dict = {
            f"model_{i}": float(w)
            for i, w in enumerate(self.weights)
        }
        return weights_dict


class EnsembleStacking:
    """
    Cross-validation stacking for robust ensemble.
    """
    
    def __init__(self, num_folds: int = 5):
        self.num_folds = num_folds
        self.meta_learners: List[MetaLearner] = []
    
    def train_stacked(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_base_models: int = 4
    ) -> np.ndarray:
        """
        Train stacked meta-learners using cross-validation.
        
        Returns: Meta-features for final meta-learner training
        """
        fold_size = len(X) // self.num_folds
        meta_features = np.zeros((len(X), num_base_models))
        
        for fold in range(self.num_folds):
            # Create train/validation split
            val_idx = slice(fold * fold_size, (fold + 1) * fold_size)
            train_idx = np.concatenate([
                np.arange(0, fold * fold_size),
                np.arange((fold + 1) * fold_size, len(X))
            ])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train meta-learner
            meta_learner = MetaLearner(num_base_models=num_base_models)
            meta_learner.train(X_train, y_train)
            self.meta_learners.append(meta_learner)
            
            # Generate meta-features for validation set
            meta_pred = X_val @ meta_learner.weights
            meta_features[val_idx] = meta_pred.reshape(-1, 1)
        
        return meta_features


class EnsemblePredictor:
    """
    Main ensemble predictor combining all base models and meta-learner.
    """
    
    def __init__(self):
        self.base_models: Dict[str, BaseModel] = {
            "rl_agent": RLAgentPredictor(),
            "regime_detector": RegimeDetectorPredictor(),
            "sentiment": SentimentPredictor(),
            "technical": TechnicalIndicatorPredictor()
        }
        
        self.meta_learner = MetaLearner(num_base_models=len(self.base_models))
        self.scaler = StandardScaler()
        
        # Performance tracking by regime
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}
        self.prediction_history: List[EnsemblePrediction] = []
    
    def predict(self, input_data: PredictionInput) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        """
        # Get predictions from base models
        base_predictions = {}
        base_predictions_array = []
        
        for name, model in self.base_models.items():
            pred = model.predict({
                "rsi": input_data.rsi,
                "macd": input_data.macd,
                "bollinger_position": input_data.bollinger_position,
                "sentiment_score": input_data.sentiment_score,
                "current_regime": self._detect_regime(input_data),
                "recent_return": (input_data.price - input_data.metadata.get("prev_price", input_data.price)) / input_data.price
            })
            base_predictions[name] = pred
            base_predictions_array.append(pred)
        
        # Meta-learner prediction
        base_predictions_array = np.array(base_predictions_array).reshape(1, -1)
        regime = self._detect_regime(input_data)
        
        ensemble_pred, confidence = self.meta_learner.predict(
            base_predictions_array[0],
            regime
        )
        
        # Generate action recommendation
        if ensemble_pred > 0.3:
            action = "BUY"
        elif ensemble_pred < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        
        result = EnsemblePrediction(
            timestamp=input_data.timestamp,
            predicted_direction=ensemble_pred,
            predicted_return=ensemble_pred * 2.0,  # Scale to percentage
            confidence=confidence,
            base_predictions=base_predictions,
            weights=self.meta_learner.get_weights(),
            regime=regime,
            recommended_action=action
        )
        
        self.prediction_history.append(result)
        
        # Track performance by regime
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                "predictions": 0,
                "correct": 0,
                "mae": 0.0
            }
        
        return result
    
    def _detect_regime(self, input_data: PredictionInput) -> MarketRegime:
        """Detect current market regime"""
        volatility = input_data.volatility
        trend = input_data.trend_strength
        
        if volatility > 0.03:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.5:
            return MarketRegime.TRENDING_UP
        elif trend < -0.5:
            return MarketRegime.TRENDING_DOWN
        elif abs(trend) < 0.2:
            return MarketRegime.RANGING
        else:
            return MarketRegime.MEAN_REVERT
    
    def train_on_history(
        self,
        predictions: List[EnsemblePrediction],
        actual_returns: List[float]
    ):
        """
        Train meta-learner on prediction history.
        """
        if len(predictions) < 10:
            logger.warning("Not enough data to train meta-learner")
            return
        
        # Extract base predictions
        X = np.array([
            [p.base_predictions.get(name, 0) for name in self.base_models.keys()]
            for p in predictions
        ])
        y = np.array(actual_returns)
        
        # Extract regimes
        regimes = [p.regime for p in predictions]
        
        # Train meta-learner
        self.meta_learner.train(X, y, regimes)
        
        logger.info(f"Trained meta-learner on {len(predictions)} historical predictions")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        if not self.prediction_history:
            return {}
        
        predictions = np.array([p.predicted_return for p in self.prediction_history])
        confidences = np.array([p.confidence for p in self.prediction_history])
        
        stats = {
            "total_predictions": len(self.prediction_history),
            "avg_confidence": float(np.mean(confidences)),
            "avg_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "buy_signals": len([p for p in self.prediction_history if p.recommended_action == "BUY"]),
            "sell_signals": len([p for p in self.prediction_history if p.recommended_action == "SELL"]),
            "hold_signals": len([p for p in self.prediction_history if p.recommended_action == "HOLD"]),
            "regime_distribution": {
                regime.value: len([p for p in self.prediction_history if p.regime == regime])
                for regime in MarketRegime
            },
            "meta_learner_weights": self.meta_learner.get_weights()
        }
        
        return stats

"""
Market Regime Detection Module

Uses Hidden Markov Models and statistical methods to detect
market regimes (bull, bear, sideways) and adapt trading strategies.

Features:
- Hidden Markov Model for regime classification
- Volatility regime detection using GARCH
- Trend/Mean-reversion detection with Hurst exponent
- Change point detection with CUSUM
- Regime-aware strategy selection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    """Current regime state with probabilities"""
    primary_regime: MarketRegime
    volatility_regime: MarketRegime
    trend_regime: MarketRegime
    regime_probabilities: dict[str, float]
    confidence: float
    duration_days: int
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "primary_regime": self.primary_regime.value,
            "volatility_regime": self.volatility_regime.value,
            "trend_regime": self.trend_regime.value,
            "probabilities": self.regime_probabilities,
            "confidence": self.confidence,
            "duration_days": self.duration_days,
            "detected_at": self.detected_at.isoformat()
        }


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.
    Implements Baum-Welch for training and Viterbi for decoding.
    """
    
    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-6):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        
        # Model parameters (initialized during fit)
        self.startprob = None  # Initial state probabilities
        self.transmat = None   # Transition matrix
        self.means = None      # Emission means
        self.covars = None     # Emission covariances
        
        self._fitted = False
    
    def _init_params(self, X: np.ndarray):
        """Initialize model parameters"""
        n_samples, n_features = X.shape
        
        # Random initialization
        self.startprob = np.ones(self.n_states) / self.n_states
        self.transmat = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # K-means style initialization for means
        indices = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X[indices]
        
        # Initialize covariances
        self.covars = np.array([np.cov(X.T) + np.eye(n_features) * 0.1 
                                for _ in range(self.n_states)])
    
    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log likelihood of observations for each state"""
        n_samples = len(X)
        log_prob = np.zeros((n_samples, self.n_states))
        
        for k in range(self.n_states):
            try:
                log_prob[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means[k], cov=self.covars[k]
                )
            except Exception:
                # Fallback to diagonal covariance
                log_prob[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means[k], cov=np.diag(np.diag(self.covars[k]))
                )
        
        return log_prob
    
    def _forward(self, log_prob: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm (alpha pass)"""
        n_samples = len(log_prob)
        log_alpha = np.zeros((n_samples, self.n_states))
        
        # Initialize
        log_alpha[0] = np.log(self.startprob + 1e-10) + log_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.transmat[:, j] + 1e-10)
                ) + log_prob[t, j]
        
        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood
    
    def _backward(self, log_prob: np.ndarray) -> np.ndarray:
        """Backward algorithm (beta pass)"""
        n_samples = len(log_prob)
        log_beta = np.zeros((n_samples, self.n_states))
        
        # Initialize (log(1) = 0)
        log_beta[-1] = 0
        
        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transmat[i] + 1e-10) + log_prob[t+1] + log_beta[t+1]
                )
        
        return log_beta
    
    def _compute_posteriors(self, log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities (gamma)"""
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)
    
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit the HMM using Baum-Welch algorithm"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self._init_params(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step
            log_prob = self._compute_log_likelihood(X)
            log_alpha, log_likelihood = self._forward(log_prob)
            log_beta = self._backward(log_prob)
            
            gamma = self._compute_posteriors(log_alpha, log_beta)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                logger.info(f"HMM converged at iteration {iteration}")
                break
            prev_log_likelihood = log_likelihood
            
            # M-step
            # Update start probabilities
            self.startprob = gamma[0] / gamma[0].sum()
            
            # Update transition matrix
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(n_samples - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j] += np.exp(
                            log_alpha[t, i] + 
                            np.log(self.transmat[i, j] + 1e-10) +
                            log_prob[t+1, j] +
                            log_beta[t+1, j] -
                            log_likelihood
                        )
            
            self.transmat = xi / (xi.sum(axis=1, keepdims=True) + 1e-10)
            
            # Update emission parameters
            for k in range(self.n_states):
                weights = gamma[:, k]
                weights_sum = weights.sum() + 1e-10
                
                self.means[k] = (weights[:, np.newaxis] * X).sum(axis=0) / weights_sum
                
                diff = X - self.means[k]
                self.covars[k] = (weights[:, np.newaxis, np.newaxis] * 
                                 (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / weights_sum
                self.covars[k] += np.eye(n_features) * 0.01  # Regularization
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence using Viterbi"""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(X)
        log_prob = self._compute_log_likelihood(X)
        
        # Viterbi algorithm
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize
        viterbi[0] = np.log(self.startprob + 1e-10) + log_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                scores = viterbi[t-1] + np.log(self.transmat[:, j] + 1e-10)
                backpointer[t, j] = np.argmax(scores)
                viterbi[t, j] = scores[backpointer[t, j]] + log_prob[t, j]
        
        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]
        
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities"""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        log_prob = self._compute_log_likelihood(X)
        log_alpha, _ = self._forward(log_prob)
        log_beta = self._backward(log_prob)
        
        return self._compute_posteriors(log_alpha, log_beta)


class VolatilityRegimeDetector:
    """Detects volatility regimes using GARCH-like modeling"""
    
    def __init__(self, lookback: int = 60, high_vol_threshold: float = 1.5, low_vol_threshold: float = 0.5):
        self.lookback = lookback
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
        self.long_term_vol = None
    
    def fit(self, returns: np.ndarray):
        """Estimate long-term volatility"""
        self.long_term_vol = np.std(returns) * np.sqrt(252)
        return self
    
    def detect(self, returns: np.ndarray) -> tuple[MarketRegime, float]:
        """Detect current volatility regime"""
        if len(returns) < self.lookback:
            return MarketRegime.SIDEWAYS, 0.5
        
        # Calculate realized volatility
        recent_returns = returns[-self.lookback:]
        realized_vol = np.std(recent_returns) * np.sqrt(252)
        
        if self.long_term_vol is None:
            self.long_term_vol = realized_vol
        
        vol_ratio = realized_vol / (self.long_term_vol + 1e-10)
        
        if vol_ratio > self.high_vol_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min((vol_ratio - self.high_vol_threshold) / self.high_vol_threshold, 1.0)
        elif vol_ratio < self.low_vol_threshold:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min((self.low_vol_threshold - vol_ratio) / self.low_vol_threshold, 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5
        
        return regime, confidence


class HurstExponentCalculator:
    """Calculate Hurst exponent for trend/mean-reversion detection"""
    
    @staticmethod
    def calculate(series: np.ndarray, max_lag: int = 100) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        H > 0.5: Trending
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        """
        if len(series) < max_lag:
            max_lag = len(series) // 2
        
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            # Calculate R/S for this lag
            n_chunks = len(series) // lag
            if n_chunks < 1:
                continue
            
            rs_chunk = []
            for i in range(n_chunks):
                chunk = series[i * lag:(i + 1) * lag]
                mean_chunk = np.mean(chunk)
                
                # Cumulative deviation
                y = np.cumsum(chunk - mean_chunk)
                r = np.max(y) - np.min(y)  # Range
                s = np.std(chunk)  # Standard deviation
                
                if s > 0:
                    rs_chunk.append(r / s)
            
            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression on log-log plot
        x = np.log([v[0] for v in rs_values])
        y = np.log([v[1] for v in rs_values])
        
        slope, _, _, _, _ = stats.linregress(x, y)
        
        return slope


class ChangePointDetector:
    """Detect structural changes using CUSUM"""
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.0):
        self.threshold = threshold
        self.drift = drift
        
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.mean = None
        self.std = None
    
    def fit(self, series: np.ndarray):
        """Fit baseline statistics"""
        self.mean = np.mean(series)
        self.std = np.std(series)
        return self
    
    def detect(self, value: float) -> tuple[bool, str]:
        """Detect if a change point occurred"""
        if self.mean is None or self.std is None:
            return False, "not_fitted"
        
        z = (value - self.mean) / (self.std + 1e-10)
        
        self.cusum_pos = max(0, self.cusum_pos + z - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - z - self.drift)
        
        if self.cusum_pos > self.threshold:
            self.cusum_pos = 0
            return True, "positive_shift"
        elif self.cusum_neg > self.threshold:
            self.cusum_neg = 0
            return True, "negative_shift"
        
        return False, "none"
    
    def reset(self):
        """Reset CUSUM accumulators"""
        self.cusum_pos = 0
        self.cusum_neg = 0


class MarketRegimeDetector:
    """
    Main regime detection system combining multiple methods.
    """
    
    def __init__(
        self,
        hmm_states: int = 3,
        lookback_days: int = 252,
        volatility_window: int = 20
    ):
        self.hmm = GaussianHMM(n_states=hmm_states)
        self.vol_detector = VolatilityRegimeDetector(lookback=volatility_window)
        self.change_detector = ChangePointDetector()
        
        self.lookback_days = lookback_days
        self.regime_history: list[RegimeState] = []
        self.current_regime_start = 0
        
        self._fitted = False
    
    def fit(self, prices: np.ndarray):
        """Fit regime detection models"""
        returns = np.diff(np.log(prices))
        
        # Fit HMM
        features = self._create_features(returns)
        self.hmm.fit(features)
        
        # Fit volatility detector
        self.vol_detector.fit(returns)
        
        # Fit change point detector
        self.change_detector.fit(returns)
        
        self._fitted = True
        return self
    
    def _create_features(self, returns: np.ndarray) -> np.ndarray:
        """Create feature matrix for HMM"""
        n = len(returns)
        
        # Simple features: returns and rolling volatility
        window = min(20, n // 4)
        
        if n < window:
            return returns.reshape(-1, 1)
        
        rolling_vol = np.array([
            np.std(returns[max(0, i-window):i+1]) if i >= window else np.std(returns[:i+1])
            for i in range(n)
        ])
        
        features = np.column_stack([returns, rolling_vol])
        return features
    
    def detect(self, prices: np.ndarray) -> RegimeState:
        """Detect current market regime"""
        if not self._fitted:
            self.fit(prices[-self.lookback_days:])
        
        returns = np.diff(np.log(prices))
        features = self._create_features(returns)
        
        # HMM prediction
        states = self.hmm.predict(features)
        probabilities = self.hmm.predict_proba(features)
        
        current_state = states[-1]
        current_probs = probabilities[-1]
        
        # Map HMM state to regime
        state_returns = {}
        for s in range(self.hmm.n_states):
            state_mask = states == s
            if state_mask.any():
                state_returns[s] = np.mean(returns[state_mask])
        
        # Sort states by average return
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        
        state_to_regime = {}
        if len(sorted_states) >= 3:
            state_to_regime[sorted_states[0][0]] = MarketRegime.BEAR
            state_to_regime[sorted_states[1][0]] = MarketRegime.SIDEWAYS
            state_to_regime[sorted_states[2][0]] = MarketRegime.BULL
        else:
            for i, (s, _) in enumerate(sorted_states):
                if i == 0:
                    state_to_regime[s] = MarketRegime.BEAR
                else:
                    state_to_regime[s] = MarketRegime.BULL
        
        primary_regime = state_to_regime.get(current_state, MarketRegime.SIDEWAYS)
        
        # Volatility regime
        vol_regime, vol_confidence = self.vol_detector.detect(returns)
        
        # Trend regime (Hurst exponent)
        hurst = HurstExponentCalculator.calculate(prices[-min(100, len(prices)):])
        if hurst > 0.6:
            trend_regime = MarketRegime.TRENDING
        elif hurst < 0.4:
            trend_regime = MarketRegime.MEAN_REVERTING
        else:
            trend_regime = MarketRegime.SIDEWAYS
        
        # Check for change points
        is_change, change_type = self.change_detector.detect(returns[-1])
        if is_change:
            logger.info(f"Change point detected: {change_type}")
            self.current_regime_start = len(prices)
        
        # Calculate duration
        duration = len(prices) - self.current_regime_start
        
        # Calculate confidence
        confidence = current_probs.max()
        
        regime_state = RegimeState(
            primary_regime=primary_regime,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            regime_probabilities={
                MarketRegime.BULL.value: float(current_probs[sorted_states[-1][0]] if len(sorted_states) > 0 else 0.33),
                MarketRegime.BEAR.value: float(current_probs[sorted_states[0][0]] if len(sorted_states) > 0 else 0.33),
                MarketRegime.SIDEWAYS.value: float(current_probs[sorted_states[1][0]] if len(sorted_states) > 2 else 0.34)
            },
            confidence=confidence,
            duration_days=duration
        )
        
        self.regime_history.append(regime_state)
        return regime_state
    
    def get_strategy_recommendation(self, regime_state: RegimeState) -> dict:
        """Get strategy recommendations based on regime"""
        recommendations = {
            "position_sizing": 1.0,
            "strategy_type": "balanced",
            "risk_level": "medium",
            "suggested_actions": []
        }
        
        # Adjust based on primary regime
        if regime_state.primary_regime == MarketRegime.BULL:
            recommendations["position_sizing"] = 1.2
            recommendations["strategy_type"] = "momentum"
            recommendations["suggested_actions"].append("Increase long exposure")
        elif regime_state.primary_regime == MarketRegime.BEAR:
            recommendations["position_sizing"] = 0.5
            recommendations["strategy_type"] = "defensive"
            recommendations["suggested_actions"].append("Reduce exposure, consider hedging")
        
        # Adjust based on volatility
        if regime_state.volatility_regime == MarketRegime.HIGH_VOLATILITY:
            recommendations["position_sizing"] *= 0.7
            recommendations["risk_level"] = "high"
            recommendations["suggested_actions"].append("Reduce position sizes, widen stops")
        elif regime_state.volatility_regime == MarketRegime.LOW_VOLATILITY:
            recommendations["suggested_actions"].append("Consider mean-reversion strategies")
        
        # Adjust based on trend regime
        if regime_state.trend_regime == MarketRegime.TRENDING:
            recommendations["strategy_type"] = "trend_following"
            recommendations["suggested_actions"].append("Use trend-following indicators")
        elif regime_state.trend_regime == MarketRegime.MEAN_REVERTING:
            recommendations["strategy_type"] = "mean_reversion"
            recommendations["suggested_actions"].append("Trade reversals at extremes")
        
        return recommendations
    
    def get_regime_statistics(self) -> dict:
        """Get statistics about regime history"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for state in self.regime_history:
            regime = state.primary_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(self.regime_history)
        regime_percentages = {k: v / total * 100 for k, v in regime_counts.items()}
        
        avg_duration = np.mean([s.duration_days for s in self.regime_history])
        avg_confidence = np.mean([s.confidence for s in self.regime_history])
        
        return {
            "regime_distribution": regime_percentages,
            "average_duration_days": avg_duration,
            "average_confidence": avg_confidence,
            "current_regime": self.regime_history[-1].to_dict() if self.regime_history else None
        }

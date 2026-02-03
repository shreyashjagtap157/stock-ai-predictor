"""
Reinforcement Learning Trading Agents

Implements RL-based trading agents using DQN and PPO algorithms
that learn optimal trading policies from market interaction.

Features:
- DQN (Deep Q-Network) agent for discrete actions
- PPO (Proximal Policy Optimization) for continuous actions
- Custom trading environment with realistic simulation
- Experience replay and prioritized sampling
- Multi-asset support with portfolio management
"""

import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. RL agents will not function.")


class TradingAction(Enum):
    """Discrete trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4


@dataclass
class TradingState:
    """State representation for trading environment"""
    prices: np.ndarray  # Historical prices
    positions: np.ndarray  # Current positions per asset
    portfolio_value: float
    cash: float
    technical_indicators: np.ndarray  # RSI, MACD, etc.
    market_features: np.ndarray  # Volume, volatility, etc.
    
    def to_tensor(self) -> np.ndarray:
        """Flatten state to numpy array"""
        return np.concatenate([
            self.prices.flatten(),
            self.positions.flatten(),
            [self.portfolio_value, self.cash],
            self.technical_indicators.flatten(),
            self.market_features.flatten()
        ])


@dataclass 
class Experience:
    """Single experience tuple for replay"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0


class ReplayBuffer:
    """Experience replay buffer with optional prioritization"""
    
    def __init__(self, capacity: int = 100000, prioritized: bool = True, alpha: float = 0.6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha  # Prioritization exponent
        
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        experience.priority = max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[list[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritization"""
        size = len(self.buffer)
        
        if self.prioritized:
            priorities = self.priorities[:size] ** self.alpha
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(size, batch_size, p=probabilities, replace=False)
            
            # Importance sampling weights
            weights = (size * probabilities[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(size, batch_size, replace=False)
            weights = np.ones(batch_size)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class TradingEnvironment:
    """
    Trading environment for RL agents.
    Simulates realistic market conditions with transaction costs.
    """
    
    def __init__(
        self,
        price_data: np.ndarray,  # Shape: (timesteps, num_assets)
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        max_position_pct: float = 0.25,  # Max 25% in single asset
        lookback: int = 30
    ):
        self.price_data = price_data
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.lookback = lookback
        
        self.num_assets = price_data.shape[1] if len(price_data.shape) > 1 else 1
        self.num_steps = len(price_data)
        
        # State dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = len(TradingAction) * self.num_assets
        
        self.reset()
    
    def _calculate_state_dim(self) -> int:
        """Calculate state dimension"""
        return (
            self.lookback * self.num_assets +  # Price history
            self.num_assets +  # Positions
            2 +  # Portfolio value, cash
            self.num_assets * 4 +  # Technical indicators (RSI, MACD, BB, ATR)
            self.num_assets * 2  # Volume, volatility
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets)
        self.portfolio_history = [self.initial_cash]
        self.trades = []
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return (state, reward, done, info)
        """
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_action(action)
        
        # Advance time
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        
        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value)
        
        self.portfolio_history.append(new_portfolio_value)
        
        info = {
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "step": self.current_step
        }
        
        return self._get_state(), reward, done, info
    
    def _execute_action(self, action: int):
        """Execute a trading action"""
        asset_idx = action // len(TradingAction)
        action_type = TradingAction(action % len(TradingAction))
        
        if asset_idx >= self.num_assets:
            return  # Invalid action
        
        current_price = self._get_current_price(asset_idx)
        
        if action_type == TradingAction.BUY:
            self._buy(asset_idx, 0.1, current_price)
        elif action_type == TradingAction.STRONG_BUY:
            self._buy(asset_idx, 0.2, current_price)
        elif action_type == TradingAction.SELL:
            self._sell(asset_idx, 0.5, current_price)
        elif action_type == TradingAction.STRONG_SELL:
            self._sell(asset_idx, 1.0, current_price)
    
    def _buy(self, asset_idx: int, fraction: float, price: float):
        """Buy asset with fraction of available cash"""
        max_investment = self.cash * fraction
        
        # Apply position limit
        portfolio_value = self._get_portfolio_value()
        max_position_value = portfolio_value * self.max_position_pct
        current_position_value = self.positions[asset_idx] * price
        allowed_investment = max(0, max_position_value - current_position_value)
        
        investment = min(max_investment, allowed_investment)
        
        if investment > 0:
            # Apply slippage and transaction cost
            effective_price = price * (1 + self.slippage)
            shares = investment / effective_price
            cost = investment * (1 + self.transaction_cost)
            
            if cost <= self.cash:
                self.positions[asset_idx] += shares
                self.cash -= cost
                self.trades.append({
                    "step": self.current_step,
                    "type": "BUY",
                    "asset": asset_idx,
                    "shares": shares,
                    "price": effective_price
                })
    
    def _sell(self, asset_idx: int, fraction: float, price: float):
        """Sell fraction of position"""
        shares_to_sell = self.positions[asset_idx] * fraction
        
        if shares_to_sell > 0:
            # Apply slippage
            effective_price = price * (1 - self.slippage)
            proceeds = shares_to_sell * effective_price
            proceeds_after_cost = proceeds * (1 - self.transaction_cost)
            
            self.positions[asset_idx] -= shares_to_sell
            self.cash += proceeds_after_cost
            self.trades.append({
                "step": self.current_step,
                "type": "SELL",
                "asset": asset_idx,
                "shares": shares_to_sell,
                "price": effective_price
            })
    
    def _get_current_price(self, asset_idx: int = 0) -> float:
        """Get current price for an asset"""
        if self.num_assets == 1:
            return self.price_data[self.current_step]
        return self.price_data[self.current_step, asset_idx]
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        if self.num_assets == 1:
            price = self.price_data[self.current_step]
            position_value = self.positions[0] * price
        else:
            prices = self.price_data[self.current_step]
            position_value = np.sum(self.positions * prices)
        
        return self.cash + position_value
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Price history
        if self.num_assets == 1:
            prices = self.price_data[self.current_step - self.lookback:self.current_step]
            prices = prices.reshape(-1, 1)
        else:
            prices = self.price_data[self.current_step - self.lookback:self.current_step]
        
        # Normalize prices
        prices_normalized = prices / prices[-1] - 1
        
        # Technical indicators (simplified)
        returns = np.diff(prices, axis=0) / prices[:-1]
        rsi = self._calculate_rsi(prices)
        volatility = np.std(returns, axis=0) * np.sqrt(252)
        
        state = TradingState(
            prices=prices_normalized,
            positions=self.positions / max(np.sum(np.abs(self.positions)), 1),
            portfolio_value=self._get_portfolio_value() / self.initial_cash,
            cash=self.cash / self.initial_cash,
            technical_indicators=np.concatenate([rsi, volatility]),
            market_features=np.zeros(self.num_assets * 2)  # Placeholder
        )
        
        return state.to_tensor()
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return np.zeros(self.num_assets)
        
        deltas = np.diff(prices, axis=0)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:], axis=0)
        avg_loss = np.mean(losses[-period:], axis=0)
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi / 100  # Normalize to 0-1
    
    def _calculate_reward(self, prev_value: float, new_value: float) -> float:
        """Calculate reward using risk-adjusted returns"""
        returns = (new_value - prev_value) / prev_value
        
        # Penalize drawdowns
        max_value = max(self.portfolio_history)
        drawdown = (max_value - new_value) / max_value
        drawdown_penalty = -drawdown * 0.5
        
        # Sharpe-like reward
        if len(self.portfolio_history) > 30:
            recent_returns = np.diff(self.portfolio_history[-30:]) / np.array(self.portfolio_history[-31:-1])
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10)
            sharpe_bonus = np.clip(sharpe, -1, 1) * 0.1
        else:
            sharpe_bonus = 0
        
        return returns + drawdown_penalty + sharpe_bonus


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network with dueling architecture"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Dueling streams
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine using dueling formula
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
    
    
    class DQNAgent:
        """
        Deep Q-Network agent for discrete trading actions.
        Uses double DQN with dueling architecture and prioritized replay.
        """
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            learning_rate: float = 1e-4,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_size: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 100,
            device: str = "auto"
        ):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            
            # Networks
            self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.buffer = ReplayBuffer(buffer_size, prioritized=True)
            
            self.train_step = 0
            self.losses = []
        
        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """Select action using epsilon-greedy policy"""
            if training and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        def store_experience(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool):
            """Store experience in replay buffer"""
            self.buffer.push(Experience(state, action, reward, next_state, done))
        
        def train_step_update(self) -> float:
            """Perform one training step"""
            if len(self.buffer) < self.batch_size:
                return 0.0
            
            # Sample batch
            experiences, indices, weights = self.buffer.sample(
                self.batch_size,
                beta=min(1.0, 0.4 + self.train_step * 0.0001)
            )
            
            # Prepare tensors
            states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
            actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
            dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN: use policy net to select action, target net to evaluate
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
            
            # Compute loss with importance sampling weights
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none').squeeze()).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities
            self.buffer.update_priorities(indices, td_errors.flatten())
            
            # Update target network
            self.train_step += 1
            if self.train_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            loss_val = loss.item()
            self.losses.append(loss_val)
            return loss_val
        
        def save(self, path: str):
            """Save agent state"""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'train_step': self.train_step
            }, path)
        
        def load(self, path: str):
            """Load agent state"""
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']


    class PPONetwork(nn.Module):
        """Actor-Critic network for PPO"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Actor (policy)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Critic (value)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            shared_features = self.shared(x)
            action_probs = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_probs, value
        
        def get_action(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
            action_probs, value = self.forward(x)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, value


    class PPOAgent:
        """
        Proximal Policy Optimization agent for trading.
        Provides stable policy updates with trust region constraint.
        """
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            epochs: int = 10,
            batch_size: int = 64,
            device: str = "auto"
        ):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.epochs = epochs
            self.batch_size = batch_size
            
            self.network = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            
            # Trajectory storage
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.dones = []
        
        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """Select action from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob, value = self.network.get_action(state_tensor)
                self.log_probs.append(log_prob)
                self.values.append(value)
                return action
            else:
                with torch.no_grad():
                    action_probs, _ = self.network(state_tensor)
                    return action_probs.argmax().item()
        
        def store_transition(self, state: np.ndarray, action: int, reward: float, done: bool):
            """Store transition for trajectory"""
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        
        def train_on_trajectory(self) -> dict:
            """Train on collected trajectory using PPO"""
            if len(self.states) == 0:
                return {}
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(self.states)).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs).detach().squeeze()
            old_values = torch.stack(self.values).detach().squeeze()
            
            # Compute advantages using GAE
            advantages = self._compute_gae(self.rewards, old_values.cpu().numpy(), self.dones)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = advantages + old_values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            total_loss = 0
            for _ in range(self.epochs):
                # Mini-batch updates
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]
                    
                    batch_states = states[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]
                    
                    # Forward pass
                    action_probs, values = self.network(batch_states)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(), batch_returns)
                    
                    # Total loss
                    loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
            
            # Clear trajectory
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.dones = []
            
            return {"loss": total_loss / self.epochs}
        
        def _compute_gae(self, rewards: list, values: np.ndarray, dones: list) -> np.ndarray:
            """Compute Generalized Advantage Estimation"""
            advantages = np.zeros(len(rewards))
            last_advantage = 0
            last_value = 0
            
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    last_advantage = 0
                    last_value = 0
                
                delta = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
                advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
                
                last_advantage = advantages[t]
                last_value = values[t]
            
            return advantages
        
        def save(self, path: str):
            """Save agent"""
            torch.save({
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        
        def load(self, path: str):
            """Load agent"""
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])


class RLTrainer:
    """Trainer for RL trading agents"""
    
    def __init__(self, env: TradingEnvironment, agent_type: str = "dqn"):
        self.env = env
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for RL training")
        
        if agent_type == "dqn":
            self.agent = DQNAgent(env.state_dim, env.action_dim)
        elif agent_type == "ppo":
            self.agent = PPOAgent(env.state_dim, env.action_dim)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agent_type = agent_type
        self.episode_rewards = []
        self.episode_values = []
    
    def train(self, num_episodes: int = 1000, log_interval: int = 10) -> dict:
        """Train the agent"""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                if self.agent_type == "dqn":
                    self.agent.store_experience(state, action, reward, next_state, done)
                    self.agent.train_step_update()
                elif self.agent_type == "ppo":
                    self.agent.store_transition(state, action, reward, done)
                
                state = next_state
                total_reward += reward
            
            if self.agent_type == "ppo":
                self.agent.train_on_trajectory()
            
            final_value = info["portfolio_value"]
            self.episode_rewards.append(total_reward)
            self.episode_values.append(final_value)
            
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_value = np.mean(self.episode_values[-log_interval:])
                logger.info(f"Episode {episode + 1}: Avg Reward = {avg_reward:.4f}, Avg Final Value = ${avg_value:.2f}")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_values": self.episode_values,
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
            "final_avg_value": np.mean(self.episode_values[-100:])
        }
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate the trained agent"""
        total_rewards = []
        final_values = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            
            total_rewards.append(total_reward)
            final_values.append(info["portfolio_value"])
        
        return {
            "avg_reward": np.mean(total_rewards),
            "avg_final_value": np.mean(final_values),
            "sharpe_ratio": np.mean(total_rewards) / (np.std(total_rewards) + 1e-10) * np.sqrt(252)
        }

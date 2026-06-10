"""
APEX Model E: PPO Position Manager

Reinforcement learning agent that decides position size, entry timing,
and exit timing.  Trained with Stable-Baselines3 PPO over a Gymnasium
environment wrapping NautilusTrader.

Observation : ~25-dim vector of model outputs + portfolio state.
Action      : Continuous Box(-1, 1) mapped to position adjustment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Gymnasium environment
# ---------------------------------------------------------------------------

class ApexTradingEnv(gymnasium.Env):
    """
    Minimal Gymnasium wrapper for APEX position management.

    This environment is used for training the PPO agent.  In production the
    agent receives live observations from the ensemble pipeline; this env
    exists only for RL training with historical replay.

    Observation space : Box(25,)  -- see OBSERVATION_KEYS
    Action space      : Box(-1, 1)  -- position adjustment fraction
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: np.ndarray,
        returns: np.ndarray,
        initial_balance: float = 10_000.0,
        transaction_cost_bps: float = 5.0,
        max_position: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        observations : (n_steps, obs_dim) array of pre-computed observation vectors.
        returns : (n_steps,) array of per-step market returns.
        initial_balance : starting cash.
        transaction_cost_bps : round-trip cost in basis points.
        max_position : max absolute position as fraction of portfolio.
        """
        super().__init__()
        self.observations = observations.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.initial_balance = initial_balance
        self.transaction_cost_bps = transaction_cost_bps
        self.max_position = max_position

        obs_dim = observations.shape[1]
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # State
        self._step_idx: int = 0
        self._position: float = 0.0
        self._balance: float = initial_balance
        self._peak_balance: float = initial_balance

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._position = 0.0
        self._balance = self.initial_balance
        self._peak_balance = self.initial_balance
        return self.observations[0], {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_position = float(np.clip(action[0], -1.0, 1.0)) * self.max_position

        # Transaction cost
        delta = abs(target_position - self._position)
        cost = delta * self._balance * (self.transaction_cost_bps / 10_000)

        # PnL from current position
        market_return = float(self.returns[self._step_idx])
        pnl = self._position * self._balance * market_return

        # Update state
        self._balance += pnl - cost
        self._position = target_position
        self._peak_balance = max(self._peak_balance, self._balance)
        self._step_idx += 1

        # Compute reward: risk-adjusted return
        step_return = (pnl - cost) / max(self.initial_balance, 1.0)
        drawdown = (self._peak_balance - self._balance) / max(self._peak_balance, 1.0)

        # Reward = return - drawdown penalty - transaction cost penalty
        reward = float(step_return - 0.5 * drawdown - 0.1 * (cost / self.initial_balance))

        terminated = self._step_idx >= len(self.observations) - 1
        truncated = self._balance <= 0

        info = {
            "balance": self._balance,
            "position": self._position,
            "pnl": pnl,
            "cost": cost,
            "drawdown": drawdown,
        }

        obs = self.observations[min(self._step_idx, len(self.observations) - 1)]
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# PPO Position Model
# ---------------------------------------------------------------------------

class PPOPositionModel:
    """
    PPO agent for position management.

    Observation keys (25-dim vector):
        xgb_edge, lgbm_return, tft_median, tft_spread,
        regime_trending_up, regime_trending_down, regime_mean_reverting,
        regime_volatile, regime_low_activity,
        current_position, unrealized_pnl, cash_balance,
        time_to_expiry, spread_bps, volume_zscore,
        portfolio_cvar, drawdown_pct, deployed_pct,
        sentiment_score, news_velocity,
        ensemble_score, edge_ci_lower, edge_ci_upper,
        regime_stability, breaker_level_encoded

    Action: continuous float in [-1, 1] mapped to position adjustment.
        +1 = go fully long, -1 = go fully short, 0 = flat.
    """

    OBSERVATION_KEYS = [
        "xgb_edge",
        "lgbm_return",
        "tft_median",
        "tft_spread",
        "regime_trending_up",
        "regime_trending_down",
        "regime_mean_reverting",
        "regime_volatile",
        "regime_low_activity",
        "current_position",
        "unrealized_pnl",
        "cash_balance",
        "time_to_expiry",
        "spread_bps",
        "volume_zscore",
        "portfolio_cvar",
        "drawdown_pct",
        "deployed_pct",
        "sentiment_score",
        "news_velocity",
        "ensemble_score",
        "edge_ci_lower",
        "edge_ci_upper",
        "regime_stability",
        "breaker_level_encoded",
    ]

    def __init__(
        self,
        policy: str = "MlpPolicy",
        net_arch: list[int] | None = None,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "auto",
    ) -> None:
        self.policy = policy
        self.net_arch = net_arch or [256, 256]
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.model: Any = None  # stable_baselines3.PPO
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env: gymnasium.Env,
        total_timesteps: int = 100_000,
        eval_env: gymnasium.Env | None = None,
        log_interval: int = 10,
    ) -> dict[str, float]:
        """
        Train PPO agent on the given environment.

        Parameters
        ----------
        env : Gymnasium environment (e.g. ``ApexTradingEnv``).
        total_timesteps : total training steps across all episodes.
        eval_env : optional separate env for evaluation.
        log_interval : episodes between logging.

        Returns
        -------
        Training summary metrics.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback

        policy_kwargs = {"net_arch": self.net_arch}

        self.model = PPO(
            policy=self.policy,
            env=env,
            learning_rate=self.learning_rate,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=self.device,
        )

        callbacks = []
        if eval_env is not None:
            eval_cb = EvalCallback(
                eval_env,
                eval_freq=max(total_timesteps // 20, 1000),
                n_eval_episodes=5,
                verbose=0,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=log_interval,
        )

        self._is_fitted = True

        # Evaluate final performance
        metrics = self._evaluate(env, n_episodes=10)
        logger.info("PPOPositionModel trained: %s", metrics)
        return metrics

    def train_from_data(
        self,
        observations: np.ndarray,
        returns: np.ndarray,
        total_timesteps: int = 100_000,
        initial_balance: float = 10_000.0,
    ) -> dict[str, float]:
        """
        Convenience method: create an ApexTradingEnv from data and train.

        Parameters
        ----------
        observations : (n_steps, 25) array of observation vectors.
        returns : (n_steps,) array of per-step market returns.
        total_timesteps : PPO training steps.
        initial_balance : starting cash for the env.
        """
        env = ApexTradingEnv(
            observations=observations,
            returns=returns,
            initial_balance=initial_balance,
        )
        return self.train(env, total_timesteps=total_timesteps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> float:
        """
        Predict position adjustment given current observation.

        Parameters
        ----------
        observation : (obs_dim,) vector of current state.
        deterministic : if True, use mean of policy distribution.

        Returns
        -------
        float in [-1, 1] representing the target position adjustment.
        """
        self._check_fitted()
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(np.clip(action[0], -1.0, 1.0))

    def predict_with_value(
        self, observation: np.ndarray,
    ) -> tuple[float, float]:
        """
        Predict action and estimated value of the current state.

        Returns
        -------
        (action, value_estimate) tuple.
        """
        self._check_fitted()
        import torch

        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        obs_tensor = torch.tensor(obs, device=self.model.device)
        with torch.no_grad():
            value = self.model.policy.predict_values(obs_tensor)
            action, _ = self.model.predict(obs, deterministic=True)

        return float(np.clip(action[0], -1.0, 1.0)), float(value.cpu().numpy()[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the trained PPO model."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Saved PPO model to %s", path)

    def load(self, path: Path) -> None:
        """Load a trained PPO model from disk."""
        from stable_baselines3 import PPO

        path = Path(path)
        self.model = PPO.load(str(path), device=self.device)
        self._is_fitted = True
        logger.info("Loaded PPO model from %s", path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        env: gymnasium.Env,
        n_episodes: int = 10,
    ) -> dict[str, float]:
        """Run evaluation episodes and compute summary statistics."""
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        final_balances: list[float] = []
        max_drawdowns: list[float] = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            peak = env.initial_balance if hasattr(env, "initial_balance") else 10_000.0
            max_dd = 0.0

            while True:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(
                    np.array([action], dtype=np.float32),
                )
                total_reward += reward
                steps += 1

                if "balance" in info:
                    peak = max(peak, info["balance"])
                    dd = (peak - info["balance"]) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            final_balances.append(info.get("balance", 0.0))
            max_drawdowns.append(max_dd)

        rewards_arr = np.array(episode_rewards)
        balances_arr = np.array(final_balances)

        # Sharpe-like ratio on episode rewards
        mean_r = float(np.mean(rewards_arr))
        std_r = float(np.std(rewards_arr)) or 1e-8
        sharpe = mean_r / std_r

        return {
            "mean_reward": mean_r,
            "std_reward": std_r,
            "sharpe_ratio": sharpe,
            "mean_final_balance": float(np.mean(balances_arr)),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }

    @staticmethod
    def build_observation(
        *,
        xgb_edge: float = 0.0,
        lgbm_return: float = 0.0,
        tft_median: float = 0.0,
        tft_spread: float = 0.0,
        regime_probs: dict[str, float] | None = None,
        current_position: float = 0.0,
        unrealized_pnl: float = 0.0,
        cash_balance: float = 0.0,
        time_to_expiry: float = 0.0,
        spread_bps: float = 0.0,
        volume_zscore: float = 0.0,
        portfolio_cvar: float = 0.0,
        drawdown_pct: float = 0.0,
        deployed_pct: float = 0.0,
        sentiment_score: float = 0.0,
        news_velocity: float = 0.0,
        ensemble_score: float = 0.0,
        edge_ci_lower: float = 0.0,
        edge_ci_upper: float = 0.0,
        regime_stability: float = 0.0,
        breaker_level_encoded: float = 0.0,
    ) -> np.ndarray:
        """
        Construct the 25-dim observation vector from named components.

        Convenience method for production code that builds observations
        from multiple model outputs.
        """
        rp = regime_probs or {}
        return np.array([
            xgb_edge,
            lgbm_return,
            tft_median,
            tft_spread,
            rp.get("trending_up", 0.0),
            rp.get("trending_down", 0.0),
            rp.get("mean_reverting", 0.0),
            rp.get("volatile", 0.0),
            rp.get("low_activity", 0.0),
            current_position,
            unrealized_pnl,
            cash_balance,
            time_to_expiry,
            spread_bps,
            volume_zscore,
            portfolio_cvar,
            drawdown_pct,
            deployed_pct,
            sentiment_score,
            news_velocity,
            ensemble_score,
            edge_ci_lower,
            edge_ci_upper,
            regime_stability,
            breaker_level_encoded,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "PPOPositionModel has not been trained. Call train() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"<PPOPositionModel [{status}, arch={self.net_arch}, "
            f"lr={self.learning_rate}]>"
        )

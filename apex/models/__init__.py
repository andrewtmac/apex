"""
APEX Model Zoo

Seven ML models + a central registry for versioning and deployment.

Models:
    A. XGBoostProbabilityModel   -- Binary probability estimator (XGBoost + isotonic)
    B. LightGBMReturnModel       -- 1h forward return predictor (LightGBM)
    C. TFTForecastModel          -- Multi-horizon quantile forecaster (Temporal Fusion Transformer)
    D. LSTMRegimeModel           -- Market regime classifier (LSTM)
    E. PPOPositionModel          -- RL position manager (PPO via SB3)
    F. FinBERTSentimentModel     -- Headline sentiment classifier (FinBERT)
    G. BayesianCalibrationModel  -- Uncertainty quantification (Beta-Binomial posterior)

Registry:
    ModelRegistry                -- Version control, persistence, promotion, rollback
"""

from apex.models.bayesian_calibration import BayesianCalibrationModel
from apex.models.finbert_sentiment import FinBERTSentimentModel
from apex.models.lgbm_return import LightGBMReturnModel
from apex.models.lstm_regime import LSTMRegimeModel
from apex.models.ppo_position import ApexTradingEnv, PPOPositionModel
from apex.models.registry import ModelRegistry, ModelVersion
from apex.models.tft_forecast import TFTForecastModel
from apex.models.xgboost_prob import XGBoostProbabilityModel

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelVersion",
    # Models A-G
    "XGBoostProbabilityModel",
    "LightGBMReturnModel",
    "TFTForecastModel",
    "LSTMRegimeModel",
    "PPOPositionModel",
    "ApexTradingEnv",
    "FinBERTSentimentModel",
    "BayesianCalibrationModel",
]

"""APEX model training pipeline.

Submodules:
- train_xgb: Daily XGBoost probability estimator retraining
- train_lgbm: Daily LightGBM return predictor retraining
- train_tft: Weekly Temporal Fusion Transformer retraining
- train_regime: Weekly LSTM regime detector retraining
- train_ppo: PPO position manager training
- train_finbert: Monthly FinBERT fine-tuning
- scheduler: Training schedule orchestrator
- validator: Pre-deployment model validation
"""

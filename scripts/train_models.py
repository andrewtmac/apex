#!/usr/bin/env python3
"""Train all APEX models on collected historical data.

Models trained:
  A. XGBoost probability estimator (on resolved markets)
  B. LightGBM return predictor (on resolved markets)
  D. LSTM regime detector (placeholder - needs time-series)
  F. FinBERT sentiment (load pretrained)
  G. Bayesian calibration (initialize from training data)

Models deferred:
  C. TFT forecast (needs live time-series data)
  E. PPO position manager (needs RL environment + backtest data)
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd
import structlog

structlog.configure(processors=[structlog.dev.ConsoleRenderer(colors=True)])
logger = structlog.get_logger()

MODELS_DIR = Path(__file__).parent.parent / "models_store"
MODELS_DIR.mkdir(exist_ok=True)


def build_features_from_resolved(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build basic feature matrix from resolved market data.

    Since we don't have real-time features for historical markets,
    we use the metadata available: final_price, volume, duration,
    category encoding, venue encoding, and derived features.
    """
    features = pd.DataFrame()

    # Price features
    features["final_price"] = df["final_price"].fillna(0.5)
    features["price_distance_from_50"] = abs(df["final_price"].fillna(0.5) - 0.5)
    features["price_near_round"] = df["final_price"].fillna(0.5).apply(
        lambda p: min(abs(p - round(p * 10) / 10), 0.05)
    )
    features["implied_prob"] = df["final_price"].fillna(0.5)
    features["log_odds"] = np.log(
        (df["final_price"].fillna(0.5).clip(0.01, 0.99)) /
        (1 - df["final_price"].fillna(0.5).clip(0.01, 0.99))
    )

    # Volume features
    vol = df["volume"].fillna(0)
    features["log_volume"] = np.log1p(vol)
    features["volume_zscore"] = (vol - vol.mean()) / (vol.std() + 1e-8)
    features["high_volume"] = (vol > vol.quantile(0.75)).astype(float)

    # Duration features
    if "duration_hours" in df.columns:
        dur = df["duration_hours"].fillna(0)
        features["log_duration_hours"] = np.log1p(dur.clip(0))
        features["short_duration"] = (dur < 24).astype(float)
        features["long_duration"] = (dur > 720).astype(float)

    # Liquidity
    if "liquidity" in df.columns:
        features["log_liquidity"] = np.log1p(df["liquidity"].fillna(0))

    # Venue encoding
    features["is_polymarket"] = (df["venue"] == "polymarket").astype(float)
    features["is_kalshi"] = (df["venue"] == "kalshi").astype(float)

    # Category encoding (top categories)
    if "category" in df.columns:
        top_cats = df["category"].value_counts().head(15).index.tolist()
        for cat in top_cats:
            safe_name = cat.replace(" ", "_").replace("-", "_").lower()[:20]
            features[f"cat_{safe_name}"] = (df["category"] == cat).astype(float)

    # Favorite-longshot bias features
    features["is_heavy_favorite"] = (df["final_price"].fillna(0.5) > 0.85).astype(float)
    features["is_heavy_underdog"] = (df["final_price"].fillna(0.5) < 0.15).astype(float)
    features["is_tossup"] = (
        (df["final_price"].fillna(0.5) > 0.4) &
        (df["final_price"].fillna(0.5) < 0.6)
    ).astype(float)

    # Clean up
    features = features.fillna(0)
    feature_names = features.columns.tolist()
    X = features.values.astype(np.float32)
    y = df["outcome"].values.astype(np.float32)

    return X, y, feature_names


async def train_xgboost(X, y, feature_names):
    """Train Model A: XGBoost Probability Estimator."""
    from apex.models.xgboost_prob import XGBoostProbabilityModel

    logger.info("train.xgboost_starting", n_samples=len(y), n_features=len(feature_names))

    model = XGBoostProbabilityModel()
    metrics = model.train(X, y, feature_names)

    # Save model
    import pickle
    model_path = MODELS_DIR / "xgboost_prob_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Export ONNX if possible
    try:
        onnx_path = MODELS_DIR / "xgboost_prob_v1.onnx"
        model.export_onnx(onnx_path)
        logger.info("train.xgboost_onnx_exported", path=str(onnx_path))
    except Exception as e:
        logger.warning("train.xgboost_onnx_failed", error=str(e))

    logger.info("train.xgboost_done", **metrics)
    return model, metrics


async def train_lightgbm(X, y, feature_names):
    """Train Model B: LightGBM Return Predictor."""
    from apex.models.lgbm_return import LightGBMReturnModel

    # For return prediction, we create pseudo-returns from outcome vs price
    # return = outcome - final_price (positive if market was underpriced)
    logger.info("train.lightgbm_starting", n_samples=len(y), n_features=len(feature_names))

    model = LightGBMReturnModel()

    # Pseudo-returns: how much the market moved from its last price to resolution
    pseudo_returns = y - X[:, 0]  # outcome - final_price

    metrics = model.train(X, pseudo_returns, feature_names)

    import pickle
    model_path = MODELS_DIR / "lgbm_return_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("train.lightgbm_done", **metrics)
    return model, metrics


async def train_bayesian_calibration(X, y, feature_names, xgb_model):
    """Train Model G: Bayesian Calibration Layer."""
    from apex.models.bayesian_calibration import BayesianCalibrationModel

    logger.info("train.calibration_starting")

    model = BayesianCalibrationModel()

    # Get XGBoost predictions for calibration
    predictions = xgb_model.predict(X)

    # Fit calibration on predictions vs actuals
    model.fit_calibration(predictions, y.astype(int))

    # Update posterior with all observations
    for pred, actual in zip(predictions, y.astype(int)):
        model.update_posterior(float(pred), int(actual))

    import pickle
    model_path = MODELS_DIR / "bayesian_calibration_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Compute calibration metrics
    calibrated = np.array([model.calibrate(p) for p in predictions])
    brier_raw = np.mean((predictions - y) ** 2)
    brier_cal = np.mean((calibrated - y) ** 2)

    metrics = {
        "brier_raw": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier_cal), 4),
        "improvement": round(float((brier_raw - brier_cal) / brier_raw * 100), 1),
        "n_observations": len(y),
    }

    logger.info("train.calibration_done", **metrics)
    return model, metrics


async def load_finbert():
    """Load Model F: FinBERT (pre-trained, no fine-tuning needed yet)."""
    from apex.models.finbert_sentiment import FinBERTSentimentModel

    logger.info("train.finbert_loading")

    model = FinBERTSentimentModel()
    model.load()

    # Test with sample headlines
    test_headlines = [
        "Bitcoin surges past $100,000 as institutional demand grows",
        "Federal Reserve signals aggressive rate cuts ahead",
        "Major earthquake strikes Tokyo, markets plunge",
        "Weather forecast shows mild temperatures for Northeast",
        "NBA Finals: Lakers dominate in Game 7 victory",
    ]

    results = model.predict_batch(test_headlines)
    for headline, result in zip(test_headlines, results):
        score = model.sentiment_score(result)
        logger.info(
            "train.finbert_test",
            headline=headline[:50],
            score=round(score, 3),
            pos=round(result["positive"], 3),
            neg=round(result["negative"], 3),
        )

    import pickle
    model_path = MODELS_DIR / "finbert_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model_name": model.model_name, "loaded": True}, f)

    logger.info("train.finbert_done")
    return model


async def main():
    db_url = os.environ.get("DATABASE_URL", "postgresql://odin-mini@localhost:5432/apex")

    print("\n" + "=" * 70)
    print("  APEX Model Training Pipeline")
    print("=" * 70 + "\n")

    t0 = time.time()

    # Load training data
    csv_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    if csv_path.exists():
        logger.info("train.loading_csv", path=str(csv_path))
        df = pd.read_csv(csv_path)
    else:
        logger.info("train.loading_db")
        import asyncpg
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM resolved_markets ORDER BY resolved_at DESC"
            )
        await pool.close()
        df = pd.DataFrame([dict(r) for r in rows])
        # Normalize column names
        df = df.rename(columns={"title": "question", "id": "market_id"})

    logger.info("train.data_loaded", n_rows=len(df), columns=list(df.columns))

    if len(df) < 100:
        logger.error("train.insufficient_data", n_rows=len(df))
        print("ERROR: Need at least 100 resolved markets to train. Run collect first.")
        return

    # Build features
    X, y, feature_names = build_features_from_resolved(df)
    logger.info(
        "train.features_built",
        n_samples=X.shape[0],
        n_features=X.shape[1],
        yes_rate=round(float(y.mean()), 3),
    )

    # === Train Model A: XGBoost ===
    print("\n--- Model A: XGBoost Probability Estimator ---")
    xgb_model, xgb_metrics = await train_xgboost(X, y, feature_names)

    # === Train Model B: LightGBM ===
    print("\n--- Model B: LightGBM Return Predictor ---")
    lgbm_model, lgbm_metrics = await train_lightgbm(X, y, feature_names)

    # === Train Model G: Bayesian Calibration ===
    print("\n--- Model G: Bayesian Calibration Layer ---")
    cal_model, cal_metrics = await train_bayesian_calibration(X, y, feature_names, xgb_model)

    # === Load Model F: FinBERT ===
    print("\n--- Model F: FinBERT Sentiment (loading pre-trained) ---")
    try:
        finbert = await load_finbert()
        finbert_status = "LOADED"
    except Exception as e:
        logger.warning("train.finbert_failed", error=str(e))
        finbert_status = f"FAILED: {e}"

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("  APEX Model Training -- Results")
    print("=" * 70)
    print(f"\n  Training data: {len(df):,} resolved markets")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training time: {elapsed:.1f}s")
    print()
    print(f"  Model A (XGBoost):     {xgb_metrics}")
    print(f"  Model B (LightGBM):    {lgbm_metrics}")
    print(f"  Model G (Calibration): {cal_metrics}")
    print(f"  Model F (FinBERT):     {finbert_status}")
    print()
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Files: {list(MODELS_DIR.glob('*.pkl'))}")
    print()
    print("  Models D (LSTM Regime), C (TFT), E (PPO) require live")
    print("  time-series data and will be trained during paper trading.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

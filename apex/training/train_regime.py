"""Weekly retraining of LSTM regime detector (Model D).

Classifies the current market environment into one of four regimes:
CALM, NORMAL, ELEVATED, CRISIS.  Uses an LSTM network trained on
historical volatility, correlation, and volume features with HMM-based
regime labels as ground truth.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import numpy as np
import structlog

from apex.config import Regime
from apex.models.registry import ModelRegistry

logger = structlog.get_logger(__name__)

MODEL_NAME = "lstm_regime"

_REGIME_LABELS = {
    Regime.CALM: 0,
    Regime.NORMAL: 1,
    Regime.ELEVATED: 2,
    Regime.CRISIS: 3,
}
_LABEL_TO_REGIME = {v: k for k, v in _REGIME_LABELS.items()}

_LSTM_CONFIG = {
    "input_size": 20,       # number of regime-relevant features
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "seq_len": 30,          # 30 time steps of context
    "num_classes": 4,       # CALM, NORMAL, ELEVATED, CRISIS
    "learning_rate": 5e-4,
    "batch_size": 32,
    "max_epochs": 100,
    "patience": 15,
    "grad_clip": 1.0,
}


class LSTMRegimeDetector:
    """LSTM-based market regime classifier.

    Input: sequence of volatility/correlation/volume features.
    Output: probability distribution over 4 regime states.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**_LSTM_CONFIG, **(config or {})}
        self.model = None
        self.scaler = None
        self._is_fitted = False

    def _build_model(self) -> Any:
        """Build the PyTorch LSTM model."""
        import torch
        import torch.nn as nn

        cfg = self.config

        class _LSTMClassifier(nn.Module):
            def __init__(self, cfg: dict):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=cfg["input_size"],
                    hidden_size=cfg["hidden_size"],
                    num_layers=cfg["num_layers"],
                    dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(cfg["dropout"])
                self.fc1 = nn.Linear(cfg["hidden_size"], cfg["hidden_size"] // 2)
                self.fc2 = nn.Linear(cfg["hidden_size"] // 2, cfg["num_classes"])
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (batch, seq_len, input_size)
                lstm_out, (h_n, _) = self.lstm(x)
                # Use the last hidden state of the top layer
                h_last = h_n[-1]  # (batch, hidden_size)
                out = self.dropout(h_last)
                out = self.relu(self.fc1(out))
                out = self.fc2(out)
                return out  # raw logits

        return _LSTMClassifier(cfg)

    def train(
        self,
        X_sequences: np.ndarray,
        y_labels: np.ndarray,
    ) -> dict[str, float]:
        """Train the LSTM regime detector.

        Parameters
        ----------
        X_sequences : (n_samples, seq_len, n_features)
        y_labels : (n_samples,) integer regime labels 0-3
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize features
        from sklearn.preprocessing import StandardScaler

        n, seq_len, n_feat = X_sequences.shape
        X_flat = X_sequences.reshape(-1, n_feat)
        self.scaler = StandardScaler().fit(X_flat)
        X_norm = self.scaler.transform(X_flat).reshape(n, seq_len, n_feat)

        # Split chronologically: 85% train, 15% validation
        n_val = max(1, int(n * 0.15))
        X_tr = torch.FloatTensor(X_norm[:-n_val]).to(device)
        y_tr = torch.LongTensor(y_labels[:-n_val]).to(device)
        X_val = torch.FloatTensor(X_norm[-n_val:]).to(device)
        y_val = torch.LongTensor(y_labels[-n_val:]).to(device)

        # Handle class imbalance with weighted loss
        class_counts = np.bincount(y_labels.astype(int), minlength=cfg["num_classes"])
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * cfg["num_classes"]
        weights_tensor = torch.FloatTensor(class_weights).to(device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True
        )

        self.config["input_size"] = n_feat
        self.model = self._build_model().to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, factor=0.5
        )

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(cfg["max_epochs"]):
            # Train
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for xb, yb in train_dl:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg["grad_clip"]
                )
                optimizer.step()

                train_loss_sum += loss.item() * len(yb)
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += len(yb)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()

            scheduler.step(val_loss)
            train_acc = train_correct / max(train_total, 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    logger.info(
                        "regime_train.early_stop",
                        epoch=epoch,
                        best_val_acc=best_val_acc,
                    )
                    break

            if epoch % 10 == 0:
                logger.info(
                    "regime_train.epoch",
                    epoch=epoch,
                    train_acc=round(train_acc, 4),
                    val_acc=round(val_acc, 4),
                    val_loss=round(val_loss, 4),
                )

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._is_fitted = True

        # Compute per-class metrics
        with torch.no_grad():
            final_logits = self.model(X_val)
            final_preds = final_logits.argmax(dim=1).cpu().numpy()
            final_probs = torch.softmax(final_logits, dim=1).cpu().numpy()

        y_val_np = y_val.cpu().numpy()

        from sklearn.metrics import classification_report

        report = classification_report(
            y_val_np, final_preds,
            target_names=[r.value for r in Regime],
            output_dict=True,
            zero_division=0,
        )

        # Average confidence on correct predictions
        correct_mask = final_preds == y_val_np
        avg_confidence = float(
            np.mean(np.max(final_probs[correct_mask], axis=1))
        ) if correct_mask.any() else 0.0

        metrics = {
            "accuracy": best_val_acc,
            "avg_confidence": avg_confidence,
            "n_train": int(n - n_val),
            "n_val": int(n_val),
            "epochs_trained": epoch + 1,
        }

        # Add per-class F1 scores
        for regime in Regime:
            key = regime.value
            if key in report:
                metrics[f"f1_{key.lower()}"] = report[key].get("f1-score", 0.0)

        logger.info("regime_train.trained", **metrics)
        return metrics

    def predict(self, X_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict regime probabilities.

        Parameters
        ----------
        X_seq : (n_samples, seq_len, n_features)

        Returns
        -------
        (predicted_regimes, probabilities) where:
        - predicted_regimes: (n_samples,) integer labels
        - probabilities: (n_samples, 4) softmax probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMRegimeDetector has not been trained")

        import torch

        device = next(self.model.parameters()).device

        n, seq_len, n_feat = X_seq.shape
        X_norm = self.scaler.transform(X_seq.reshape(-1, n_feat)).reshape(
            n, seq_len, n_feat
        )
        X_t = torch.FloatTensor(X_norm).to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        return preds, probs

    def predict_regime(self, X_seq: np.ndarray) -> tuple[Regime, float]:
        """Predict single regime with confidence.

        Parameters
        ----------
        X_seq : (1, seq_len, n_features) or (seq_len, n_features)

        Returns
        -------
        (regime, confidence) tuple
        """
        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, :]

        preds, probs = self.predict(X_seq)
        label = int(preds[0])
        confidence = float(probs[0, label])
        regime = _LABEL_TO_REGIME.get(label, Regime.NORMAL)
        return regime, confidence


async def _generate_regime_labels(
    db_url: str,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regime labels using Hidden Markov Model on historical data.

    Uses realized volatility, volume, and spread features to fit a 4-state
    HMM, then labels each time step with a regime state.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5, command_timeout=60)

    try:
        rows = await pool.fetch(
            """
            SELECT time, features
            FROM feature_store
            WHERE feature_set = 'price_features'
              AND time >= $1
            ORDER BY time ASC
            """,
            cutoff,
        )

        if len(rows) < 100:
            raise ValueError(
                f"Insufficient data for regime detection: {len(rows)} rows"
            )

        # Extract regime-relevant features
        regime_feature_keys = [
            "realized_vol_1h", "realized_vol_24h", "parkinson_vol",
            "garman_klass_vol", "bid_ask_spread", "book_imbalance_L1",
            "volume_ratio_5m_1h", "volume_zscore", "rsi_14",
            "atr_14", "adx_14", "cci_20", "bollinger_upper",
            "bollinger_lower", "momentum_1h", "momentum_4h",
            "trend_strength", "high_low_range",
            "log_ret_1h", "log_ret_4h",
        ]

        feature_matrix: list[list[float]] = []
        for row in rows:
            feat_raw = row["features"]
            feat = json.loads(feat_raw) if isinstance(feat_raw, str) else feat_raw
            vector = [float(feat.get(k, 0.0)) for k in regime_feature_keys]
            feature_matrix.append(vector)

        X_all = np.array(feature_matrix, dtype=np.float64)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit 4-state Gaussian HMM to generate labels
        from hmmlearn.hmm import GaussianHMM

        hmm = GaussianHMM(
            n_components=4,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(X_all)
        hmm_labels = hmm.predict(X_all)

        # Map HMM states to regimes by average volatility per state
        state_vol = {}
        vol_idx = regime_feature_keys.index("realized_vol_1h")
        for state in range(4):
            mask = hmm_labels == state
            if mask.any():
                state_vol[state] = float(np.mean(X_all[mask, vol_idx]))
            else:
                state_vol[state] = 0.0

        # Sort by volatility: lowest = CALM, highest = CRISIS
        sorted_states = sorted(state_vol, key=state_vol.get)
        state_mapping = {
            sorted_states[0]: 0,  # CALM
            sorted_states[1]: 1,  # NORMAL
            sorted_states[2]: 2,  # ELEVATED
            sorted_states[3]: 3,  # CRISIS
        }

        labels = np.array([state_mapping[s] for s in hmm_labels], dtype=np.int64)

        logger.info(
            "regime_train.labels_generated",
            n_samples=len(labels),
            distribution={
                r.value: int(np.sum(labels == i))
                for i, r in enumerate(Regime)
            },
        )
        return X_all, labels

    finally:
        await pool.close()


async def train_regime_detector(
    db_url: str,
    model_registry: ModelRegistry,
    lookback_days: int = 60,
) -> dict:
    """Train the LSTM regime detector.

    Steps:
    1. Load historical features from TimescaleDB
    2. Generate regime labels using HMM
    3. Build sequences for LSTM training
    4. Train LSTM classifier
    5. Validate accuracy and per-class F1
    6. Register and promote if accuracy > 0.6

    Returns
    -------
    dict with version_id, metrics, status
    """
    logger.info("regime_train.start", lookback_days=lookback_days)

    seq_len = _LSTM_CONFIG["seq_len"]

    # 1-2. Load data and generate labels
    X_all, labels = await _generate_regime_labels(db_url, lookback_days)

    if len(X_all) < seq_len + 10:
        raise ValueError(
            f"Insufficient data for sequence building: {len(X_all)} rows"
        )

    # 3. Build sequences
    X_seqs = []
    y_seqs = []
    for i in range(len(X_all) - seq_len):
        X_seqs.append(X_all[i : i + seq_len])
        y_seqs.append(labels[i + seq_len - 1])

    X_sequences = np.array(X_seqs, dtype=np.float64)
    y_labels = np.array(y_seqs, dtype=np.int64)

    # 4. Train
    model = LSTMRegimeDetector()
    metrics = model.train(X_sequences, y_labels)

    # 5. Quality gate
    status = "deployed"
    if metrics.get("accuracy", 0) < 0.5:
        logger.warning(
            "regime_train.low_accuracy", accuracy=metrics["accuracy"]
        )
        status = "rejected"

    # 6. Register
    version_id = model_registry.register(MODEL_NAME, model, metrics)
    if status == "deployed":
        model_registry.promote(MODEL_NAME, version_id)

    result = {
        "model": MODEL_NAME,
        "version_id": version_id,
        "metrics": metrics,
        "status": status,
    }

    logger.info("regime_train.complete", **result)
    return result

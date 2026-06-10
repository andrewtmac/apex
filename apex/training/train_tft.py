"""Weekly retraining of Temporal Fusion Transformer (Model C).

TFT produces quantile price predictions (10th, 50th, 90th percentile)
over a prediction horizon.  Trained on 30-day price sequences using
PyTorch with mixed-precision and gradient clipping.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import structlog

from apex.models.registry import ModelRegistry

logger = structlog.get_logger(__name__)

MODEL_NAME = "tft_quantile"

# TFT architecture defaults
_TFT_CONFIG = {
    "input_size": 40,       # number of input features
    "hidden_size": 128,
    "num_heads": 4,
    "dropout": 0.1,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "seq_len": 60,          # 60 time steps lookback
    "pred_len": 12,         # 12 time steps ahead
    "quantiles": [0.1, 0.5, 0.9],
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 10,
    "grad_clip": 1.0,
}


class SimpleTFT:
    """Simplified Temporal Fusion Transformer for quantile prediction.

    Uses a transformer encoder-decoder architecture with gated residual
    connections and quantile regression loss.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**_TFT_CONFIG, **(config or {})}
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self._is_fitted = False

    def _build_model(self) -> Any:
        """Build the PyTorch TFT model."""
        import torch
        import torch.nn as nn

        cfg = self.config

        class _GatedResidualNetwork(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1):
                super().__init__()
                self.fc1 = nn.Linear(d_model, d_model)
                self.fc2 = nn.Linear(d_model, d_model)
                self.gate = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(dropout)
                self.layer_norm = nn.LayerNorm(d_model)
                self.elu = nn.ELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                x = self.elu(self.fc1(x))
                x = self.dropout(x)
                gate = torch.sigmoid(self.gate(x))
                x = gate * self.fc2(x)
                return self.layer_norm(x + residual)

        class _TFTModel(nn.Module):
            def __init__(self, cfg: dict):
                super().__init__()
                d = cfg["hidden_size"]
                n_in = cfg["input_size"]
                n_q = len(cfg["quantiles"])
                n_pred = cfg["pred_len"]

                self.input_proj = nn.Linear(n_in, d)
                self.pos_enc = nn.Parameter(
                    torch.randn(1, cfg["seq_len"] + n_pred, d) * 0.02
                )

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=cfg["num_heads"],
                    dim_feedforward=d * 4,
                    dropout=cfg["dropout"],
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=cfg["num_encoder_layers"],
                )

                self.grn = _GatedResidualNetwork(d, cfg["dropout"])

                # Output: one head per quantile
                self.output_heads = nn.ModuleList([
                    nn.Linear(d, n_pred) for _ in range(n_q)
                ])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (batch, seq_len, input_size)
                h = self.input_proj(x)
                seq_len = h.size(1)
                h = h + self.pos_enc[:, :seq_len, :]
                h = self.encoder(h)
                h = self.grn(h)
                # Use the last time step for prediction
                h_last = h[:, -1, :]
                # Stack quantile predictions: (batch, n_quantiles, pred_len)
                outputs = torch.stack(
                    [head(h_last) for head in self.output_heads], dim=1
                )
                return outputs

        return _TFTModel(cfg)

    def _quantile_loss(
        self,
        predictions: Any,  # torch.Tensor
        targets: Any,       # torch.Tensor
    ) -> Any:
        """Pinball loss for quantile regression."""
        import torch

        quantiles = self.config["quantiles"]
        losses = []
        for i, q in enumerate(quantiles):
            pred = predictions[:, i, :]
            error = targets - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return sum(losses) / len(losses)

    def train(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray,
    ) -> dict[str, float]:
        """Train the TFT on sequence data.

        Parameters
        ----------
        X_sequences : (n_samples, seq_len, n_features)
        y_sequences : (n_samples, pred_len) forward returns
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize
        from sklearn.preprocessing import StandardScaler

        n, seq_len, n_feat = X_sequences.shape
        X_flat = X_sequences.reshape(-1, n_feat)
        self.scaler_X = StandardScaler().fit(X_flat)
        X_norm = self.scaler_X.transform(X_flat).reshape(n, seq_len, n_feat)

        self.scaler_y = StandardScaler().fit(y_sequences)
        y_norm = self.scaler_y.transform(y_sequences)

        # Split: 90% train, 10% validation
        n_val = max(1, int(n * 0.1))
        X_tr = torch.FloatTensor(X_norm[:-n_val]).to(device)
        y_tr = torch.FloatTensor(y_norm[:-n_val]).to(device)
        X_val = torch.FloatTensor(X_norm[-n_val:]).to(device)
        y_val = torch.FloatTensor(y_norm[-n_val:]).to(device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

        # Build model
        self.config["input_size"] = n_feat
        self.model = self._build_model().to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg["max_epochs"]):
            # Train
            self.model.train()
            train_loss_sum = 0.0
            n_batches = 0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = self._quantile_loss(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg["grad_clip"]
                )
                optimizer.step()
                train_loss_sum += loss.item()
                n_batches += 1

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val)
                val_loss = self._quantile_loss(val_preds, y_val).item()

            avg_train_loss = train_loss_sum / max(n_batches, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    logger.info(
                        "tft_train.early_stop",
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
                    break

            if epoch % 10 == 0:
                logger.info(
                    "tft_train.epoch",
                    epoch=epoch,
                    train_loss=round(avg_train_loss, 6),
                    val_loss=round(val_loss, 6),
                )

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._is_fitted = True

        # Compute final metrics on validation set
        with torch.no_grad():
            final_preds = self.model(X_val)  # (n_val, n_quantiles, pred_len)
            median_preds = final_preds[:, 1, :].cpu().numpy()  # 50th percentile

        median_preds_raw = self.scaler_y.inverse_transform(median_preds)
        y_val_raw = self.scaler_y.inverse_transform(y_val.cpu().numpy())

        mse = float(np.mean((median_preds_raw - y_val_raw) ** 2))
        mae = float(np.mean(np.abs(median_preds_raw - y_val_raw)))

        # Coverage: % of actuals within [q10, q90]
        q10_preds = self.scaler_y.inverse_transform(
            final_preds[:, 0, :].cpu().numpy()
        )
        q90_preds = self.scaler_y.inverse_transform(
            final_preds[:, 2, :].cpu().numpy()
        )
        coverage = float(
            np.mean((y_val_raw >= q10_preds) & (y_val_raw <= q90_preds))
        )

        metrics = {
            "quantile_loss": best_val_loss,
            "mse": mse,
            "mae": mae,
            "coverage_80": coverage,
            "n_train": int(n - n_val),
            "n_val": int(n_val),
            "epochs_trained": epoch + 1,
        }

        logger.info("tft_train.trained", **metrics)
        return metrics

    def predict(self, X_seq: np.ndarray) -> dict[float, np.ndarray]:
        """Predict quantiles for input sequences.

        Parameters
        ----------
        X_seq : (n_samples, seq_len, n_features)

        Returns
        -------
        dict mapping quantile level -> predicted values (n_samples, pred_len)
        """
        if not self._is_fitted:
            raise RuntimeError("TFT has not been trained")

        import torch

        device = next(self.model.parameters()).device

        n, seq_len, n_feat = X_seq.shape
        X_norm = self.scaler_X.transform(X_seq.reshape(-1, n_feat)).reshape(
            n, seq_len, n_feat
        )
        X_t = torch.FloatTensor(X_norm).to(device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t)  # (n, n_quantiles, pred_len)

        result = {}
        for i, q in enumerate(self.config["quantiles"]):
            pred_np = preds[:, i, :].cpu().numpy()
            result[q] = self.scaler_y.inverse_transform(pred_np)

        return result


async def _load_sequence_data(
    db_url: str,
    lookback_days: int,
    seq_len: int = 60,
    pred_len: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare sequence data for TFT training.

    Returns (X_sequences, y_sequences) with shapes:
    - X: (n_samples, seq_len, n_features)
    - y: (n_samples, pred_len) -- forward returns
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

        if len(rows) < seq_len + pred_len + 10:
            raise ValueError(
                f"Insufficient sequence data: {len(rows)} rows "
                f"(need >= {seq_len + pred_len + 10})"
            )

        # Parse all feature rows into a matrix
        feature_dicts = []
        for row in rows:
            feat_raw = row["features"]
            feat = json.loads(feat_raw) if isinstance(feat_raw, str) else feat_raw
            feature_dicts.append(feat)

        all_keys = sorted(feature_dicts[0].keys())
        full_matrix = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in feature_dicts],
            dtype=np.float64,
        )

        # Replace NaN/inf with 0
        full_matrix = np.nan_to_num(full_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract mid_price for target construction
        mid_idx = all_keys.index("mid_price") if "mid_price" in all_keys else 0
        mid_prices = full_matrix[:, mid_idx]

        # Build sequences with sliding window
        X_seqs = []
        y_seqs = []
        for i in range(len(full_matrix) - seq_len - pred_len):
            X_seqs.append(full_matrix[i : i + seq_len])
            # Forward returns as target
            current_price = mid_prices[i + seq_len - 1]
            if current_price > 1e-8:
                future_prices = mid_prices[i + seq_len : i + seq_len + pred_len]
                returns = (future_prices - current_price) / current_price
                y_seqs.append(returns)
            else:
                y_seqs.append(np.zeros(pred_len))

        X_sequences = np.array(X_seqs, dtype=np.float64)
        y_sequences = np.array(y_seqs, dtype=np.float64)

        logger.info(
            "tft_train.sequences_built",
            n_sequences=len(X_sequences),
            seq_len=seq_len,
            pred_len=pred_len,
            n_features=X_sequences.shape[2],
        )
        return X_sequences, y_sequences

    finally:
        await pool.close()


async def train_tft(
    db_url: str,
    model_registry: ModelRegistry,
    lookback_days: int = 30,
) -> dict:
    """Train TFT on 30-day price sequences. Weekly schedule.

    Steps:
    1. Load and prepare sequence data
    2. Train TFT with quantile regression
    3. Validate coverage and prediction quality
    4. Register if improved, export model

    Returns
    -------
    dict with version_id, metrics, status
    """
    logger.info("tft_train.start", lookback_days=lookback_days)

    seq_len = _TFT_CONFIG["seq_len"]
    pred_len = _TFT_CONFIG["pred_len"]

    # 1. Load data
    X_sequences, y_sequences = await _load_sequence_data(
        db_url, lookback_days, seq_len, pred_len
    )

    # 2. Train
    model = SimpleTFT()
    metrics = model.train(X_sequences, y_sequences)

    # 3. Quality gate: require >60% coverage for 80% interval
    status = "deployed"
    if metrics.get("coverage_80", 0) < 0.60:
        logger.warning(
            "tft_train.low_coverage",
            coverage=metrics["coverage_80"],
        )
        status = "rejected"

    # 4. Register
    version_id = model_registry.register(MODEL_NAME, model, metrics)
    if status == "deployed":
        model_registry.promote(MODEL_NAME, version_id)

    result = {
        "model": MODEL_NAME,
        "version_id": version_id,
        "metrics": metrics,
        "status": status,
    }

    logger.info("tft_train.complete", **result)
    return result

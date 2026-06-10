"""
APEX Model C: Temporal Fusion Transformer (TFT)

Multi-horizon probabilistic forecasting.  Predicts quantiles at 5 min,
15 min, 1 h, and 4 h horizons.  Simplified PyTorch implementation -- no
dependency on pytorch-forecasting.

Architecture:
    - Variable selection networks (GRN-based)
    - LSTM encoder / decoder
    - Interpretable multi-head attention
    - Quantile loss function

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," 2020.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GatedLinearUnit(nn.Module):
    """GLU activation: element-wise product of two linear projections."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN: context-aware non-linear processing with skip connections.
    Core building block of TFT.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim is not None
            else None
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dims don't match
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.skip(x)

        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)

        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Learns which input variables are most relevant at each time step.
    Produces per-variable weights via softmax over GRN outputs.
    """

    def __init__(
        self,
        input_dim: int,
        num_variables: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.var_dim = input_dim // num_variables

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(self.var_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_variables)
        ])

        # Flattened GRN for variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_variables, dropout, context_dim,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)  or  (batch, input_dim)
        has_seq = x.dim() == 3

        # Variable selection weights
        weights = torch.softmax(self.selection_grn(x, context), dim=-1)  # (..., num_vars)

        # Split input into per-variable chunks and process
        chunks = torch.chunk(x, self.num_variables, dim=-1)
        var_outputs = []
        for i, (chunk, grn) in enumerate(zip(chunks, self.var_grns)):
            var_outputs.append(grn(chunk))

        # Stack and weight
        stacked = torch.stack(var_outputs, dim=-1)  # (..., hidden, num_vars)
        weighted = (stacked * weights.unsqueeze(-2)).sum(dim=-1)  # (..., hidden)

        return weighted


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.
    Uses additive attention and shares value weights across heads.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, self.d_k)  # shared across heads
        self.out_proj = nn.Linear(self.d_k, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = query.shape

        # Project queries and keys per head
        Q = self.W_q(query).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value)  # (batch, seq, d_k)  -- shared

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Average attention across heads for interpretability
        attn_avg = attn.mean(dim=1)  # (batch, seq, seq)
        context = torch.matmul(attn_avg, V)  # (batch, seq, d_k)
        output = self.out_proj(context)

        return output, attn_avg


# ---------------------------------------------------------------------------
# Full TFT Module
# ---------------------------------------------------------------------------

class TFTModule(nn.Module):
    """
    Simplified Temporal Fusion Transformer.

    Input : (batch, encoder_len, input_dim)
    Output: (batch, num_horizons, num_quantiles)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        attention_heads: int = 4,
        lstm_layers: int = 2,
        num_horizons: int = 4,
        quantiles: list[float] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_horizons = num_horizons
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        num_q = len(self.quantiles)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Variable selection
        self.vsn = VariableSelectionNetwork(
            hidden_size, min(input_dim, hidden_size), hidden_size, dropout,
        )

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Static enrichment (simplified: learned context vector)
        self.static_context = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.static_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )

        # Temporal self-attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, attention_heads, dropout,
        )
        self.attn_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Position-wise feed-forward
        self.ff_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

        # Output projection: per-horizon, per-quantile
        self.horizon_proj = nn.Linear(hidden_size, num_horizons * hidden_size)
        self.output_proj = nn.Linear(hidden_size, num_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        (batch, num_horizons, num_quantiles)
        """
        batch = x.shape[0]

        # Input embedding
        h = self.input_proj(x)  # (batch, seq, hidden)

        # LSTM encoder
        enc_out, (h_n, c_n) = self.lstm_encoder(h)

        # Static enrichment
        static = self.static_context.expand(batch, -1, -1)  # (batch, 1, hidden)
        enriched = self.static_grn(enc_out)

        # Temporal self-attention
        attn_out, attn_weights = self.attention(enriched, enriched, enriched)
        attn_out = self.attn_norm(attn_out + enriched)
        attn_out = self.attn_grn(attn_out)

        # Feed-forward
        ff_out = self.ff_grn(attn_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        # Aggregate temporal dimension: use last hidden state
        last_hidden = ff_out[:, -1, :]  # (batch, hidden)

        # Project to horizons
        horizon_h = self.horizon_proj(last_hidden)  # (batch, num_horizons * hidden)
        horizon_h = horizon_h.view(batch, self.num_horizons, self.hidden_size)

        # Quantile outputs
        output = self.output_proj(horizon_h)  # (batch, num_horizons, num_quantiles)

        return output


# ---------------------------------------------------------------------------
# Quantile Loss
# ---------------------------------------------------------------------------

class QuantileLoss(nn.Module):
    """Pinball / quantile loss for multiple quantile levels."""

    def __init__(self, quantiles: list[float]) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : (batch, num_horizons, num_quantiles)
        targets : (batch, num_horizons) or (batch, num_horizons, 1)
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[..., i]
            target = targets[..., 0]
            error = target - pred_q
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.unsqueeze(-1))

        return torch.cat(losses, dim=-1).mean()


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class TFTForecastModel:
    """
    Multi-horizon probabilistic forecasting model.

    Predicts quantiles at 5 min, 15 min, 1 h, 4 h horizons.

    Usage
    -----
    >>> model = TFTForecastModel(input_dim=20)
    >>> metrics = model.train(sequences, targets)
    >>> quantile_preds = model.predict(new_sequence)
    """

    HORIZONS = ["5m", "15m", "1h", "4h"]
    DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

    def __init__(
        self,
        input_dim: int = 20,
        hidden_size: int = 128,
        attention_heads: int = 4,
        lstm_layers: int = 2,
        quantiles: list[float] | None = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.input_dim = input_dim
        self.config = {
            "input_dim": input_dim,
            "hidden_size": hidden_size,
            "attention_heads": attention_heads,
            "lstm_layers": lstm_layers,
            "quantiles": self.quantiles,
            "dropout": dropout,
            "learning_rate": learning_rate,
        }

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: TFTModule = TFTModule(
            input_dim=input_dim,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            lstm_layers=lstm_layers,
            num_horizons=len(self.HORIZONS),
            quantiles=self.quantiles,
            dropout=dropout,
        ).to(self.device)

        self.criterion = QuantileLoss(self.quantiles)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_fraction: float = 0.1,
        patience: int = 10,
    ) -> dict[str, float]:
        """
        Train TFT on time-series data.

        Parameters
        ----------
        sequences : (n_samples, seq_len, input_dim) -- input windows.
        targets   : (n_samples, num_horizons) -- target values at each horizon.
        epochs : max training epochs.
        batch_size : mini-batch size.
        val_fraction : held-out validation fraction.
        patience : early stopping patience.

        Returns
        -------
        Training metrics dict.
        """
        # Train/val split (temporal: last N samples for validation)
        n_val = max(1, int(len(sequences) * val_fraction))
        X_train = torch.tensor(sequences[:-n_val], dtype=torch.float32)
        y_train = torch.tensor(targets[:-n_val], dtype=torch.float32)
        X_val = torch.tensor(sequences[-n_val:], dtype=torch.float32)
        y_val = torch.tensor(targets[-n_val:], dtype=torch.float32)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(X_train)

            # Validation
            val_loss = self._evaluate(X_val, y_val)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                logger.info(
                    "TFT epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                    epoch, epochs, epoch_loss, val_loss,
                )

            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True

        # Final metrics
        final_val_loss = self._evaluate(X_val, y_val)
        coverage = self._compute_coverage(X_val, y_val)

        metrics = {
            "quantile_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "epochs_trained": epoch + 1,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            **{f"coverage_{q}": float(c) for q, c in zip(self.quantiles, coverage)},
        }

        logger.info("TFTForecastModel trained: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> dict[str, dict[float, float]]:
        """
        Predict quantiles for each horizon.

        Parameters
        ----------
        sequence : (seq_len, input_dim) or (1, seq_len, input_dim).

        Returns
        -------
        Nested dict: horizon_name -> {quantile: value}.
        Example: {"5m": {0.1: -0.02, 0.25: -0.01, 0.5: 0.01, ...}, ...}
        """
        self._check_fitted()
        self.model.eval()

        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]

        x = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(x)  # (1, num_horizons, num_quantiles)

        preds_np = preds.cpu().numpy()[0]  # (num_horizons, num_quantiles)

        result: dict[str, dict[float, float]] = {}
        for h_idx, horizon in enumerate(self.HORIZONS):
            result[horizon] = {
                q: float(preds_np[h_idx, q_idx])
                for q_idx, q in enumerate(self.quantiles)
            }

        return result

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Batch prediction.

        Parameters
        ----------
        sequences : (n_samples, seq_len, input_dim).

        Returns
        -------
        np.ndarray of shape (n_samples, num_horizons, num_quantiles).
        """
        self._check_fitted()
        self.model.eval()

        x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(x)

        return preds.cpu().numpy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            y_dev = y.to(self.device)
            preds = self.model(X_dev)
            loss = self.criterion(preds, y_dev)
        self.model.train()
        return float(loss.item())

    def _compute_coverage(
        self, X: torch.Tensor, y: torch.Tensor,
    ) -> list[float]:
        """Compute empirical coverage for each quantile level."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.to(self.device)).cpu().numpy()
        y_np = y.numpy()

        coverages = []
        for q_idx, q in enumerate(self.quantiles):
            below = (y_np <= preds[..., q_idx]).mean()
            coverages.append(below)

        return coverages

    def save(self, path: Path) -> None:
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "is_fitted": self._is_fitted,
            },
            path,
        )
        logger.info("Saved TFT model to %s", path)

    def load(self, path: Path) -> None:
        """Load model weights and config."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]

        # Rebuild model with saved config
        self.model = TFTModule(
            input_dim=self.config["input_dim"],
            hidden_size=self.config["hidden_size"],
            attention_heads=self.config["attention_heads"],
            lstm_layers=self.config["lstm_layers"],
            num_horizons=len(self.HORIZONS),
            quantiles=self.config["quantiles"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_fitted = checkpoint["is_fitted"]
        logger.info("Loaded TFT model from %s", path)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "TFTForecastModel has not been trained. Call train() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"<TFTForecastModel [{status}, "
            f"hidden={self.config['hidden_size']}, "
            f"heads={self.config['attention_heads']}, "
            f"horizons={self.HORIZONS}]>"
        )

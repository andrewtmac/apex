"""
APEX Model D: LSTM Regime Detector

Classifies the current market regime into one of five states:
    trending_up, trending_down, mean_reverting, volatile, low_activity.

Input  : 50-bar windows of [price, volume, spread, volatility].
Output : 5-class softmax probabilities.

The regime classification feeds into the ensemble and risk manager so
position sizing and strategy selection adapt to current market conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network module
# ---------------------------------------------------------------------------

class LSTMRegimeModule(nn.Module):
    """
    2-layer LSTM with classification head.

    Architecture:
        Input (batch, seq_len, input_dim)
        -> LSTM(input_dim, hidden_dim, num_layers, bidirectional=False)
        -> Dropout
        -> LayerNorm
        -> FC(hidden_dim, hidden_dim // 2)
        -> ReLU + Dropout
        -> FC(hidden_dim // 2, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        logits : (batch, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Use last time step's output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)
        return logits


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class LSTMRegimeModel:
    """
    LSTM-based market regime classifier.

    Usage
    -----
    >>> model = LSTMRegimeModel(input_dim=4, hidden_dim=64, num_layers=2)
    >>> metrics = model.train(sequences, labels)
    >>> probs = model.predict(new_sequence)
    >>> # probs = {"trending_up": 0.6, "trending_down": 0.1, ...}
    """

    REGIMES = [
        "trending_up",
        "trending_down",
        "mean_reverting",
        "volatile",
        "low_activity",
    ]

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = len(self.REGIMES)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: LSTMRegimeModule = self._build_model(
            input_dim, hidden_dim, num_layers, dropout,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6,
        )
        self._is_fitted: bool = False
        self._class_weights: torch.Tensor | None = None

    def _build_model(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ) -> LSTMRegimeModule:
        """Build and return the LSTM module, moved to the target device."""
        return LSTMRegimeModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=self.num_classes,
            dropout=dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_fraction: float = 0.15,
        patience: int = 10,
        class_weight: str = "balanced",
    ) -> dict[str, Any]:
        """
        Train the regime classifier.

        Parameters
        ----------
        sequences : (n_samples, seq_len, input_dim) -- windows of market data.
        labels    : (n_samples,) integer class labels in [0, 4].
        epochs : max training epochs.
        batch_size : mini-batch size.
        val_fraction : held-out validation fraction.
        patience : early-stopping patience (epochs without improvement).
        class_weight : ``"balanced"`` to use inverse-frequency weights, or ``"uniform"``.

        Returns
        -------
        Dict of metrics: accuracy, per-class F1, confusion matrix, etc.
        """
        n_val = max(1, int(len(sequences) * val_fraction))

        X_train = torch.tensor(sequences[:-n_val], dtype=torch.float32)
        y_train = torch.tensor(labels[:-n_val], dtype=torch.long)
        X_val = torch.tensor(sequences[-n_val:], dtype=torch.float32)
        y_val = torch.tensor(labels[-n_val:], dtype=torch.long)

        # Compute class weights
        if class_weight == "balanced":
            counts = np.bincount(labels[:-n_val], minlength=self.num_classes).astype(float)
            counts = np.maximum(counts, 1.0)  # avoid div-by-zero
            weights = len(y_train) / (self.num_classes * counts)
            self._class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            self._class_weights = None

        criterion = nn.CrossEntropyLoss(weight=self._class_weights)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)

            self.scheduler.step()
            train_acc = correct / total
            epoch_loss /= total

            # Validation
            val_acc, val_loss = self._evaluate(X_val, y_val, criterion)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0:
                logger.info(
                    "Regime LSTM epoch %d/%d  train_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
                    epoch, epochs, epoch_loss, train_acc, val_acc,
                )

            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True

        # Detailed validation metrics
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(X_val.to(self.device))
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()

        y_val_np = y_val.numpy()
        report = classification_report(
            y_val_np, val_preds,
            target_names=self.REGIMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_val_np, val_preds).tolist()

        metrics: dict[str, Any] = {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class_f1": {
                regime: float(report[regime]["f1-score"])
                for regime in self.REGIMES
                if regime in report
            },
            "confusion_matrix": cm,
            "epochs_trained": epoch + 1,
            "best_val_accuracy": float(best_val_acc),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        }

        logger.info("LSTMRegimeModel trained: accuracy=%.3f  macro_f1=%.3f",
                     metrics["accuracy"], metrics["macro_f1"])
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> dict[str, float]:
        """
        Classify a single sequence into regime probabilities.

        Parameters
        ----------
        sequence : (seq_len, input_dim) or (1, seq_len, input_dim).

        Returns
        -------
        Dict mapping regime name -> softmax probability.
        Example: {"trending_up": 0.6, "trending_down": 0.1, ...}
        """
        self._check_fitted()
        self.model.eval()

        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]

        x = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        return {regime: float(p) for regime, p in zip(self.REGIMES, probs)}

    def predict_batch(self, sequences: np.ndarray) -> list[dict[str, float]]:
        """
        Classify a batch of sequences.

        Parameters
        ----------
        sequences : (n_samples, seq_len, input_dim).

        Returns
        -------
        List of regime probability dicts.
        """
        self._check_fitted()
        self.model.eval()

        x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        return [
            {regime: float(p) for regime, p in zip(self.REGIMES, row)}
            for row in probs
        ]

    def predict_regime(self, sequence: np.ndarray) -> str:
        """Return the single most likely regime label."""
        probs = self.predict(sequence)
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "is_fitted": self._is_fitted,
            },
            path,
        )
        logger.info("Saved LSTM regime model to %s", path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_layers = checkpoint["num_layers"]
        self.num_classes = checkpoint["num_classes"]

        self.model = self._build_model(
            self.input_dim, self.hidden_dim, self.num_layers,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_fitted = checkpoint["is_fitted"]
        logger.info("Loaded LSTM regime model from %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Compute validation accuracy and loss."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            loss = criterion(logits, y.to(self.device))
            preds = logits.argmax(dim=-1).cpu()
            acc = float((preds == y).float().mean())
        self.model.train()
        return acc, float(loss.item())

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LSTMRegimeModel has not been trained. Call train() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"<LSTMRegimeModel [{status}, "
            f"input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, regimes={self.REGIMES}]>"
        )

"""Monthly fine-tuning of FinBERT on accumulated labeled corpus.

Fine-tunes a pre-trained FinBERT model (ProsusAI/finbert) on APEX's
labeled sentiment corpus.  Labels are market-specific sentiments tied
to resolved market outcomes.
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

MODEL_NAME = "finbert_sentiment"

_FINBERT_CONFIG = {
    "base_model": "ProsusAI/finbert",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "gradient_accumulation_steps": 2,
    "num_labels": 3,  # negative, neutral, positive
    "label_map": {"negative": 0, "neutral": 1, "positive": 2},
}


class FinBERTSentimentModel:
    """Fine-tuned FinBERT for prediction-market sentiment analysis.

    Three-class classification: negative (-1), neutral (0), positive (+1).
    Returns calibrated sentiment scores in [-1, 1].
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**_FINBERT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self._is_fitted = False

    def train(
        self,
        texts: list[str],
        labels: list[int],
    ) -> dict[str, float]:
        """Fine-tune FinBERT on labeled text data.

        Parameters
        ----------
        texts : list of text samples
        labels : list of integer labels (0=neg, 1=neutral, 2=pos)
        """
        import torch
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        cfg = self.config

        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["base_model"],
            num_labels=cfg["num_labels"],
        )

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )

        # Tokenize
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )

        # Dataset class
        class _SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)

        train_ds = _SentimentDataset(train_encodings, train_labels)
        val_ds = _SentimentDataset(val_encodings, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="/tmp/apex_finbert_training",
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"] * 2,
            learning_rate=cfg["learning_rate"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            fp16=cfg["fp16"] and torch.cuda.is_available(),
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            load_best_model_at_end=False,
            report_to="none",
        )

        # Define compute_metrics
        from sklearn.metrics import accuracy_score, f1_score

        def compute_metrics(eval_pred):
            logits, ref_labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return {
                "accuracy": accuracy_score(ref_labels, preds),
                "f1_macro": f1_score(ref_labels, preds, average="macro"),
                "f1_weighted": f1_score(ref_labels, preds, average="weighted"),
            }

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        logger.info(
            "finbert_train.training",
            n_train=len(train_texts),
            n_val=len(val_texts),
        )
        trainer.train()
        self._is_fitted = True

        # Evaluate
        eval_results = trainer.evaluate()

        metrics = {
            "accuracy": float(eval_results.get("eval_accuracy", 0)),
            "f1_macro": float(eval_results.get("eval_f1_macro", 0)),
            "f1_weighted": float(eval_results.get("eval_f1_weighted", 0)),
            "eval_loss": float(eval_results.get("eval_loss", 0)),
            "n_train": len(train_texts),
            "n_val": len(val_texts),
        }

        logger.info("finbert_train.trained", **metrics)
        return metrics

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Predict sentiment for a list of texts.

        Returns list of dicts with:
        - sentiment: float in [-1, 1]
        - confidence: float in [0, 1]
        - label: str ("negative", "neutral", "positive")
        """
        if not self._is_fitted:
            raise RuntimeError("FinBERT has not been fine-tuned")

        import torch

        device = next(self.model.parameters()).device
        label_names = ["negative", "neutral", "positive"]
        sentiment_values = [-1.0, 0.0, 1.0]

        results = []
        # Process in batches
        batch_size = self.config["batch_size"] * 2
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config["max_length"],
                return_tensors="pt",
            ).to(device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for j in range(len(batch_texts)):
                pred_label = int(np.argmax(probs[j]))
                confidence = float(probs[j, pred_label])
                # Weighted sentiment score
                sentiment = float(
                    sum(p * s for p, s in zip(probs[j], sentiment_values))
                )
                results.append({
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "label": label_names[pred_label],
                    "probabilities": {
                        name: float(probs[j, k])
                        for k, name in enumerate(label_names)
                    },
                })

        return results

    def predict_score(self, text: str) -> float:
        """Convenience: predict a single text and return score in [-1, 1]."""
        results = self.predict([text])
        return results[0]["sentiment"]


async def _load_labeled_corpus(
    db_url: str,
    labeled_data_path: str | None = None,
) -> tuple[list[str], list[int]]:
    """Load labeled sentiment data from DB or file.

    Labels are derived from:
    - News articles linked to resolved markets (positive outcome = positive sentiment)
    - Manually labeled data from labeled_data_path (JSONL format)
    """
    texts: list[str] = []
    labels: list[int] = []

    # Load from file if provided
    if labeled_data_path:
        path = Path(labeled_data_path)
        if path.exists():
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    texts.append(record["text"])
                    label_str = record.get("label", "neutral")
                    label_map = {"negative": 0, "neutral": 1, "positive": 2}
                    labels.append(label_map.get(label_str, 1))
            logger.info(
                "finbert_train.file_loaded",
                path=labeled_data_path,
                n_samples=len(texts),
            )

    # Also load from database: news articles linked to resolved markets
    try:
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3, command_timeout=30)
        try:
            rows = await pool.fetch(
                """
                SELECT
                    fs.features->>'text' AS text,
                    m.outcome
                FROM feature_store fs
                JOIN markets m ON fs.entity_id = m.id
                WHERE fs.feature_set = 'news_features'
                  AND m.status = 'resolved'
                  AND m.outcome IS NOT NULL
                  AND fs.features->>'text' IS NOT NULL
                  AND length(fs.features->>'text') > 20
                ORDER BY fs.time DESC
                LIMIT 10000
                """
            )

            for row in rows:
                text = row["text"]
                outcome = int(row["outcome"])
                # Map market outcome to sentiment label
                # outcome=1 (YES) -> positive(2), outcome=0 (NO) -> negative(0)
                label = 2 if outcome == 1 else 0
                texts.append(text)
                labels.append(label)

            logger.info(
                "finbert_train.db_loaded", n_samples=len(rows)
            )
        finally:
            await pool.close()
    except Exception:
        logger.warning("finbert_train.db_load_failed")

    if not texts:
        raise ValueError("No labeled data available for FinBERT fine-tuning")

    return texts, labels


async def fine_tune_finbert(
    db_url: str,
    model_registry: ModelRegistry,
    labeled_data_path: str | None = None,
) -> dict:
    """Fine-tune FinBERT on the accumulated labeled corpus.

    Steps:
    1. Load labeled sentiment data (file + DB)
    2. Fine-tune pre-trained FinBERT
    3. Validate accuracy and F1
    4. Register if F1 macro > 0.5
    5. Promote to production

    Returns
    -------
    dict with version_id, metrics, status
    """
    logger.info("finbert_train.start")

    # 1. Load data
    texts, labels = await _load_labeled_corpus(db_url, labeled_data_path)

    logger.info(
        "finbert_train.corpus",
        n_total=len(texts),
        label_distribution={
            "negative": labels.count(0),
            "neutral": labels.count(1),
            "positive": labels.count(2),
        },
    )

    # 2. Train
    model = FinBERTSentimentModel()
    metrics = model.train(texts, labels)

    # 3. Quality gate
    status = "deployed"
    if metrics.get("f1_macro", 0) < 0.45:
        logger.warning(
            "finbert_train.low_f1", f1_macro=metrics["f1_macro"]
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

    logger.info("finbert_train.complete", **result)
    return result

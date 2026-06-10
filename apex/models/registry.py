"""
APEX Model Registry

Central model management: versioning, persistence, validation, promotion,
rollback, and ONNX export orchestration. Every model trained in APEX is
registered here so we can reproduce results and swap models atomically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------

class ModelVersion:
    """Immutable snapshot of a registered model version."""

    __slots__ = (
        "model_name",
        "version_id",
        "timestamp",
        "metrics",
        "artifact_path",
        "is_production",
        "checksum",
    )

    def __init__(
        self,
        model_name: str,
        version_id: str,
        timestamp: str,
        metrics: dict[str, float],
        artifact_path: str,
        is_production: bool = False,
        checksum: str = "",
    ) -> None:
        self.model_name = model_name
        self.version_id = version_id
        self.timestamp = timestamp
        self.metrics = metrics
        self.artifact_path = artifact_path
        self.is_production = is_production
        self.checksum = checksum

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "is_production": self.is_production,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelVersion:
        return cls(**d)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Manages model versions, loading, validation, and deployment.

    Directory layout under *store_path*::

        models_store/
          xgboost_prob/
            manifest.json          # list of ModelVersion dicts
            v_20260609_143000/
              model.pkl
              model.onnx  (optional)
          lgbm_return/
            ...
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(self, store_path: Path = Path("models_store")) -> None:
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)

    # -- internal helpers ---------------------------------------------------

    def _model_dir(self, model_name: str) -> Path:
        return self.store_path / model_name

    def _manifest_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / self.MANIFEST_FILE

    def _load_manifest(self, model_name: str) -> list[ModelVersion]:
        path = self._manifest_path(model_name)
        if not path.exists():
            return []
        with open(path) as fh:
            data = json.load(fh)
        return [ModelVersion.from_dict(d) for d in data]

    def _save_manifest(self, model_name: str, versions: list[ModelVersion]) -> None:
        path = self._manifest_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump([v.to_dict() for v in versions], fh, indent=2)

    @staticmethod
    def _compute_checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    @staticmethod
    def _make_version_id() -> str:
        return "v_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # -- public API ---------------------------------------------------------

    def register(
        self,
        model_name: str,
        model: Any,
        metrics: dict[str, float],
    ) -> str:
        """
        Persist a model and its training metrics.

        Parameters
        ----------
        model_name : logical name, e.g. ``"xgboost_prob"``.
        model : any pickle-serialisable model object.
        metrics : training / validation metrics (Sharpe, accuracy, Brier, ...).

        Returns
        -------
        version_id : str  e.g. ``"v_20260609_143000"``.
        """
        version_id = self._make_version_id()
        version_dir = self._model_dir(model_name) / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / "model.pkl"
        payload = pickle.dumps(model)
        checksum = self._compute_checksum(payload)
        artifact_path.write_bytes(payload)

        versions = self._load_manifest(model_name)
        version = ModelVersion(
            model_name=model_name,
            version_id=version_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            artifact_path=str(artifact_path),
            is_production=False,
            checksum=checksum,
        )
        versions.append(version)
        self._save_manifest(model_name, versions)

        logger.info(
            "Registered %s %s  metrics=%s  checksum=%s",
            model_name, version_id, metrics, checksum,
        )
        return version_id

    def load(self, model_name: str, version: str = "latest") -> Any:
        """
        Load a model by name and version.

        Parameters
        ----------
        version : ``"latest"`` (newest), ``"production"`` (promoted), or an
                  explicit version id like ``"v_20260609_143000"``.
        """
        versions = self._load_manifest(model_name)
        if not versions:
            raise FileNotFoundError(f"No versions registered for '{model_name}'")

        if version == "latest":
            target = versions[-1]
        elif version == "production":
            prod = [v for v in versions if v.is_production]
            if not prod:
                raise FileNotFoundError(f"No production version for '{model_name}'")
            target = prod[-1]
        else:
            matches = [v for v in versions if v.version_id == version]
            if not matches:
                raise FileNotFoundError(
                    f"Version '{version}' not found for '{model_name}'"
                )
            target = matches[0]

        artifact = Path(target.artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact missing: {artifact}")

        payload = artifact.read_bytes()
        checksum = self._compute_checksum(payload)
        if checksum != target.checksum:
            raise ValueError(
                f"Checksum mismatch for {model_name} {target.version_id}: "
                f"expected {target.checksum}, got {checksum}"
            )

        logger.info("Loaded %s %s", model_name, target.version_id)
        return pickle.loads(payload)

    def validate(
        self,
        model_name: str,
        version: str,
        test_data: np.ndarray,
        test_labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Validate a model against held-out data.

        For models that expose a ``predict`` method, this computes:
        - MSE (always)
        - accuracy, Brier score (if test_labels are binary 0/1)

        Returns a dict of metric name -> value.
        """
        model = self.load(model_name, version)
        if not hasattr(model, "predict"):
            raise TypeError(f"Model '{model_name}' has no predict() method")

        preds = model.predict(test_data)
        preds = np.asarray(preds).ravel()

        results: dict[str, float] = {}

        if test_labels is not None:
            labels = np.asarray(test_labels).ravel()
            results["mse"] = float(np.mean((preds - labels) ** 2))

            # Binary classification metrics
            if set(np.unique(labels)).issubset({0, 1, 0.0, 1.0}):
                results["brier_score"] = float(np.mean((preds - labels) ** 2))
                binary_preds = (preds >= 0.5).astype(int)
                results["accuracy"] = float(np.mean(binary_preds == labels))

        logger.info("Validated %s %s -> %s", model_name, version, results)
        return results

    def promote(self, model_name: str, version: str) -> None:
        """
        Promote *version* to production, demoting any prior production version.
        """
        versions = self._load_manifest(model_name)
        found = False
        for v in versions:
            if v.version_id == version:
                v.is_production = True
                found = True
            else:
                v.is_production = False

        if not found:
            raise FileNotFoundError(
                f"Version '{version}' not found for '{model_name}'"
            )

        self._save_manifest(model_name, versions)
        logger.info("Promoted %s %s to production", model_name, version)

    def rollback(self, model_name: str) -> None:
        """
        Rollback to the previous production version.

        Strategy: find the current production version, demote it, and promote
        the version immediately before it.
        """
        versions = self._load_manifest(model_name)
        prod_idx: int | None = None
        for i, v in enumerate(versions):
            if v.is_production:
                prod_idx = i
                break

        if prod_idx is None:
            raise RuntimeError(f"No production version to rollback for '{model_name}'")

        if prod_idx == 0:
            raise RuntimeError(
                f"Cannot rollback '{model_name}': production version is the earliest"
            )

        versions[prod_idx].is_production = False
        versions[prod_idx - 1].is_production = True
        self._save_manifest(model_name, versions)
        logger.info(
            "Rolled back %s from %s to %s",
            model_name,
            versions[prod_idx].version_id,
            versions[prod_idx - 1].version_id,
        )

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        """Return metadata for every registered version."""
        return [v.to_dict() for v in self._load_manifest(model_name)]

    def delete_version(self, model_name: str, version: str) -> None:
        """Remove a non-production version and its artifacts."""
        versions = self._load_manifest(model_name)
        target = [v for v in versions if v.version_id == version]
        if not target:
            raise FileNotFoundError(
                f"Version '{version}' not found for '{model_name}'"
            )
        if target[0].is_production:
            raise RuntimeError("Cannot delete the production version. Rollback first.")

        artifact_dir = Path(target[0].artifact_path).parent
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        versions = [v for v in versions if v.version_id != version]
        self._save_manifest(model_name, versions)
        logger.info("Deleted %s %s", model_name, version)

    def export_onnx(
        self,
        model_name: str,
        version: str = "production",
        output_path: Path | None = None,
    ) -> Path:
        """
        Export the model to ONNX format.

        The model must implement an ``export_onnx(path)`` method.
        Returns the path to the exported ``.onnx`` file.
        """
        model = self.load(model_name, version)
        if not hasattr(model, "export_onnx"):
            raise TypeError(
                f"Model '{model_name}' does not support ONNX export"
            )

        if output_path is None:
            versions = self._load_manifest(model_name)
            matched = [v for v in versions if v.version_id == version or (
                version in ("latest", "production") and v.is_production
            )]
            if matched:
                output_path = Path(matched[-1].artifact_path).parent / "model.onnx"
            else:
                output_path = self._model_dir(model_name) / "model.onnx"

        model.export_onnx(output_path)
        logger.info("Exported ONNX for %s %s -> %s", model_name, version, output_path)
        return output_path

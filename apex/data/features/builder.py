"""
APEX Observation Builder

Central feature-engineering pipeline that orchestrates all feature extractors,
assembles a fixed-length observation vector, and applies online normalisation
via Welford's algorithm.

Usage::

    builder = ApexObservationBuilder(extractors=[
        MarketFeatureExtractor(),
        PredictionFeatureExtractor(),
        ...
    ])
    obs = await builder.build("market-123", "polymarket", raw_data)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """Maps feature names to fixed vector positions.

    The registry is populated once on first call to :meth:`register_names` and
    is immutable afterwards so that all components agree on vector layout.
    """

    def __init__(self) -> None:
        self._name_to_idx: dict[str, int] = {}
        self._idx_to_name: dict[int, str] = {}
        self._frozen: bool = False

    # -- mutation -----------------------------------------------------------

    def register_names(self, names: list[str]) -> None:
        """Append *names* to the registry.  Must be called before freeze."""
        if self._frozen:
            raise RuntimeError("Cannot register names after the registry has been frozen.")
        for name in names:
            if name in self._name_to_idx:
                raise ValueError(f"Duplicate feature name: {name}")
            idx = len(self._name_to_idx)
            self._name_to_idx[name] = idx
            self._idx_to_name[idx] = name

    def freeze(self) -> None:
        self._frozen = True

    # -- lookup -------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._name_to_idx)

    def index(self, name: str) -> int:
        return self._name_to_idx[name]

    def name(self, idx: int) -> str:
        return self._idx_to_name[idx]

    def names(self) -> list[str]:
        return [self._idx_to_name[i] for i in range(self.size)]

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __len__(self) -> int:
        return self.size


# Global singleton – shared by builder and all extractors.
FEATURE_REGISTRY = FeatureRegistry()


# ---------------------------------------------------------------------------
# Abstract base class for extractors
# ---------------------------------------------------------------------------


class FeatureExtractor(ABC):
    """Base class for all feature-group extractors.

    Subclasses must implement :meth:`feature_names` (the ordered list of
    feature names this extractor is responsible for) and :meth:`extract`.
    """

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return ordered list of feature names produced by this extractor."""
        ...

    @abstractmethod
    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        """Compute features and return ``{name: value}`` mapping.

        Missing or invalid values should be set to ``0.0`` (the normaliser
        will centre them to the running mean later).
        """
        ...


# ---------------------------------------------------------------------------
# Welford online normalizer
# ---------------------------------------------------------------------------


class WelfordNormalizer:
    """Online mean/variance normalisation using Welford's algorithm.

    Maintains per-feature running statistics and z-score-normalises each
    incoming vector.  Numerical stability is achieved via the compensated
    Welford update.

    Parameters
    ----------
    dim : int
        Dimensionality (set automatically on first call to :meth:`normalize`).
    min_samples : int
        Number of observations before normalisation kicks in (before that,
        raw values are returned).
    clip : float
        Absolute z-score clip bound to prevent extreme outliers.
    ema_halflife : int | None
        If set, applies exponential-moving-average weighting so that older
        observations decay.  ``None`` means equal weighting (standard Welford).
    """

    def __init__(
        self,
        dim: int = 0,
        min_samples: int = 30,
        clip: float = 10.0,
        ema_halflife: int | None = None,
    ) -> None:
        self.dim = dim
        self.min_samples = min_samples
        self.clip = clip
        self.ema_halflife = ema_halflife

        # Running statistics (initialised lazily)
        self.n: int = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None  # sum of squared deviations

    # -- internal -----------------------------------------------------------

    def _init_state(self, dim: int) -> None:
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    # -- public API ---------------------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Ingest a new observation to update running statistics."""
        if self.mean is None:
            self._init_state(x.shape[0])
        assert self.mean is not None and self.m2 is not None

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.mean is None or self.n < 2:
            return np.ones(self.dim, dtype=np.float64)
        assert self.m2 is not None
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Update statistics then return z-score-normalised vector."""
        self.update(x)
        if self.n < self.min_samples:
            return x.copy()
        assert self.mean is not None
        z = (x - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip)

    def inverse_normalize(self, z: np.ndarray) -> np.ndarray:
        """Map a z-score vector back to raw scale (useful for debugging)."""
        if self.mean is None:
            return z.copy()
        return z * self.std + self.mean

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "m2": self.m2.tolist() if self.m2 is not None else None,
            "dim": self.dim,
            "min_samples": self.min_samples,
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> "WelfordNormalizer":
        inst = cls(dim=d["dim"], min_samples=d["min_samples"], clip=d["clip"])
        inst.n = d["n"]
        if d["mean"] is not None:
            inst.mean = np.array(d["mean"], dtype=np.float64)
            inst.m2 = np.array(d["m2"], dtype=np.float64)
        return inst


# ---------------------------------------------------------------------------
# Redis cache helpers
# ---------------------------------------------------------------------------


@dataclass
class _CacheConfig:
    """Tiny helper wrapping optional Redis caching."""

    redis_client: Any | None = None  # redis.asyncio.Redis
    ttl_seconds: int = 60

    async def get(self, key: str) -> np.ndarray | None:
        if self.redis_client is None:
            return None
        try:
            raw = await self.redis_client.get(key)
            if raw is not None:
                return np.frombuffer(raw, dtype=np.float64).copy()
        except Exception:
            logger.debug("Redis cache miss for %s", key)
        return None

    async def set(self, key: str, arr: np.ndarray) -> None:
        if self.redis_client is None:
            return
        try:
            await self.redis_client.set(key, arr.tobytes(), ex=self.ttl_seconds)
        except Exception:
            logger.debug("Redis cache set failed for %s", key)


def _cache_key(market_id: str, venue: str, raw_data: dict[str, Any]) -> str:
    """Deterministic cache key from inputs."""
    h = hashlib.sha256()
    h.update(market_id.encode())
    h.update(venue.encode())
    # Hash a sorted JSON of non-array data to keep keys stable.
    try:
        h.update(json.dumps(raw_data, sort_keys=True, default=str).encode())
    except Exception:
        h.update(str(id(raw_data)).encode())
    return f"apex:obs:{h.hexdigest()[:24]}"


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


class ApexObservationBuilder:
    """Builds the full 165+ feature observation vector for a market.

    Parameters
    ----------
    feature_extractors : list[FeatureExtractor]
        Ordered list of extractors.  Their ``feature_names()`` are registered
        in order to define the vector layout.
    normalizer : WelfordNormalizer | None
        Online normaliser instance.  If ``None`` a fresh one is created.
    redis_client : optional
        An ``redis.asyncio.Redis`` instance for caching computed vectors.
    cache_ttl : int
        Cache time-to-live in seconds (default 60).
    """

    def __init__(
        self,
        feature_extractors: list[FeatureExtractor],
        normalizer: WelfordNormalizer | None = None,
        redis_client: Any | None = None,
        cache_ttl: int = 60,
    ) -> None:
        self.extractors = feature_extractors
        self._cache = _CacheConfig(redis_client=redis_client, ttl_seconds=cache_ttl)

        # Build registry from extractors
        for ext in self.extractors:
            names = ext.feature_names()
            FEATURE_REGISTRY.register_names(names)
        FEATURE_REGISTRY.freeze()

        self.normalizer = normalizer or WelfordNormalizer(dim=FEATURE_REGISTRY.size)
        self._dim = FEATURE_REGISTRY.size
        logger.info("ObservationBuilder ready – %d features registered.", self._dim)

    # -- public API ---------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    async def build(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> np.ndarray:
        """Build a normalised feature vector from raw market data.

        Returns
        -------
        np.ndarray
            Shape ``(dim,)`` float64 z-score normalised observation.
        """
        # 1. Cache check
        cache_key = _cache_key(market_id, venue, raw_data)
        cached = await self._cache.get(cache_key)
        if cached is not None and cached.shape[0] == self._dim:
            return cached

        # 2. Run all extractors
        features: dict[str, float] = {}
        for extractor in self.extractors:
            try:
                result = await extractor.extract(market_id, venue, raw_data)
                features.update(result)
            except Exception:
                logger.exception(
                    "Extractor %s failed for market=%s venue=%s",
                    extractor.__class__.__name__,
                    market_id,
                    venue,
                )
                # Fill with zeros for robustness
                for name in extractor.feature_names():
                    features.setdefault(name, 0.0)

        # 3. Assemble vector
        vector = self._to_vector(features)

        # 4. Normalise
        normed = self.normalizer.normalize(vector)

        # 5. Cache result
        await self._cache.set(cache_key, normed)

        return normed

    async def build_raw(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        """Like :meth:`build` but returns the un-normalised feature dict."""
        features: dict[str, float] = {}
        for extractor in self.extractors:
            try:
                result = await extractor.extract(market_id, venue, raw_data)
                features.update(result)
            except Exception:
                logger.exception("Extractor %s failed", extractor.__class__.__name__)
                for name in extractor.feature_names():
                    features.setdefault(name, 0.0)
        return features

    # -- internal helpers ---------------------------------------------------

    def _to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Pack feature dict into a fixed-length numpy vector."""
        vec = np.zeros(self._dim, dtype=np.float64)
        for name, value in features.items():
            if name in FEATURE_REGISTRY:
                idx = FEATURE_REGISTRY.index(name)
                vec[idx] = value if np.isfinite(value) else 0.0
        return vec

    def vector_to_dict(self, vector: np.ndarray) -> dict[str, float]:
        """Unpack a vector back into a named dict (debugging / logging)."""
        return {
            FEATURE_REGISTRY.name(i): float(vector[i])
            for i in range(self._dim)
        }

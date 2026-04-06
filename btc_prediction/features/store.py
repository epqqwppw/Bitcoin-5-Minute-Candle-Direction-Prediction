from __future__ import annotations

"""In-memory feature store with TTL eviction."""

import time

import pandas as pd


class FeatureStore:
    """Dict-backed feature store with time-to-live eviction.

    Provides an interface resembling a Redis feature store while keeping
    everything in-process for simplicity.

    Parameters
    ----------
    ttl_seconds : float
        Time-to-live in seconds for stored features (default 300).
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._store: dict[str, tuple[pd.DataFrame, float]] = {}
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove entries older than *ttl_seconds*."""
        now = time.time()
        expired = [
            k for k, (_, ts) in self._store.items() if now - ts > self.ttl_seconds
        ]
        for k in expired:
            del self._store[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_features(
        self, key: str, features: pd.DataFrame, timestamp: float
    ) -> None:
        """Store a features DataFrame under *key*.

        Parameters
        ----------
        key : str
            Unique identifier for this feature set.
        features : pd.DataFrame
            Feature data to store.
        timestamp : float
            Unix epoch timestamp associated with the features.
        """
        self._store[key] = (features.copy(), timestamp)

    def get_features(self, key: str) -> pd.DataFrame | None:
        """Retrieve features for *key*, or ``None`` if missing/expired.

        Triggers eviction of stale entries before lookup.
        """
        self._evict_expired()
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry[0].copy()

    def get_latest_features(self, keys: list[str]) -> dict[str, pd.DataFrame]:
        """Retrieve the latest features for a list of keys.

        Keys that are missing or expired are silently omitted.
        """
        self._evict_expired()
        result: dict[str, pd.DataFrame] = {}
        for key in keys:
            entry = self._store.get(key)
            if entry is not None:
                result[key] = entry[0].copy()
        return result

    def clear(self) -> None:
        """Remove all stored features."""
        self._store.clear()

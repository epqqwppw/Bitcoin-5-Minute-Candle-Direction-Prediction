from __future__ import annotations

"""Cross-asset feature computation module."""

import numpy as np
import pandas as pd


class CrossAssetFeatures:
    """Static methods for computing cross-asset / macro features."""

    @staticmethod
    def compute_rolling_correlation(
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 100,
    ) -> pd.Series:
        """Compute rolling Pearson correlation between two series."""
        return series_a.rolling(window=window, min_periods=1).corr(series_b)

    @staticmethod
    def compute_dynamic_correlation_matrix(
        asset_returns: pd.DataFrame, window: int = 100
    ) -> np.ndarray:
        """Compute rolling correlation matrix for multiple assets.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            Each column is an asset's return series.
        window : int
            Rolling window size.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(n_time, n_assets, n_assets)``.
        """
        n_time, n_assets = asset_returns.shape
        result = np.full((n_time, n_assets, n_assets), np.nan)
        values = asset_returns.values.astype(np.float64)

        for i in range(window - 1, n_time):
            chunk = values[i - window + 1 : i + 1]
            # Handle constant columns to avoid NaN correlations
            stds = np.nanstd(chunk, axis=0) + 1e-10
            normed = (chunk - np.nanmean(chunk, axis=0)) / stds
            corr = np.dot(normed.T, normed) / (window - 1 + 1e-10)
            # Clip to valid correlation range
            np.clip(corr, -1.0, 1.0, out=corr)
            result[i] = corr
        return result

    @staticmethod
    def compute_btc_dominance(
        btc_mcap: pd.Series, total_mcap: pd.Series
    ) -> pd.Series:
        """Compute BTC market-cap dominance ratio."""
        return btc_mcap / (total_mcap + 1e-10)

    @staticmethod
    def compute_basis_spread(
        spot_price: pd.Series, futures_price: pd.Series
    ) -> pd.Series:
        """Compute futures-spot basis spread."""
        return (futures_price - spot_price) / (spot_price + 1e-10)

    @staticmethod
    def compute_funding_rate_momentum(
        funding_rates: pd.Series, window: int = 24
    ) -> pd.Series:
        """Compute rolling sum of funding rates as a momentum signal."""
        return funding_rates.rolling(window=window, min_periods=1).sum()

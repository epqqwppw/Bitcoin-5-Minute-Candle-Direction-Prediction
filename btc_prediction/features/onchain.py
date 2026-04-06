from __future__ import annotations

"""On-chain feature computation module."""

import numpy as np
import pandas as pd


class OnChainFeatures:
    """Static methods for computing Bitcoin on-chain metrics."""

    @staticmethod
    def compute_sopr(
        spent_output_values: np.ndarray, creation_values: np.ndarray
    ) -> np.ndarray:
        """Compute Spent Output Profit Ratio.

        SOPR = spent_output_values / creation_values.
        """
        spent_output_values = np.asarray(spent_output_values, dtype=np.float64)
        creation_values = np.asarray(creation_values, dtype=np.float64)
        return spent_output_values / (creation_values + 1e-10)

    @staticmethod
    def compute_mvrv(
        market_cap: np.ndarray, realized_cap: np.ndarray
    ) -> np.ndarray:
        """Compute Market Value to Realized Value ratio."""
        market_cap = np.asarray(market_cap, dtype=np.float64)
        realized_cap = np.asarray(realized_cap, dtype=np.float64)
        return market_cap / (realized_cap + 1e-10)

    @staticmethod
    def compute_mvrv_zscore(
        mvrv: np.ndarray, window: int = 365
    ) -> np.ndarray:
        """Compute rolling z-score of MVRV."""
        mvrv = np.asarray(mvrv, dtype=np.float64)
        result = np.full_like(mvrv, np.nan)
        for i in range(window - 1, len(mvrv)):
            chunk = mvrv[i - window + 1 : i + 1]
            mu = np.nanmean(chunk)
            sigma = np.nanstd(chunk) + 1e-10
            result[i] = (mvrv[i] - mu) / sigma
        return result

    @staticmethod
    def compute_nvt(
        market_cap: np.ndarray, transaction_volume: np.ndarray
    ) -> np.ndarray:
        """Compute Network Value to Transactions ratio."""
        market_cap = np.asarray(market_cap, dtype=np.float64)
        transaction_volume = np.asarray(transaction_volume, dtype=np.float64)
        return market_cap / (transaction_volume + 1e-10)

    @staticmethod
    def compute_nvt_signal(
        market_cap: np.ndarray,
        transaction_volume: np.ndarray,
        window: int = 90,
    ) -> np.ndarray:
        """Compute NVT Signal (smoothed NVT using rolling mean of tx volume)."""
        market_cap = np.asarray(market_cap, dtype=np.float64)
        transaction_volume = np.asarray(transaction_volume, dtype=np.float64)
        result = np.full_like(market_cap, np.nan)
        for i in range(window - 1, len(market_cap)):
            avg_vol = np.nanmean(transaction_volume[i - window + 1 : i + 1])
            result[i] = market_cap[i] / (avg_vol + 1e-10)
        return result

    @staticmethod
    def compute_exchange_flow_ratio(
        inflow: np.ndarray, outflow: np.ndarray
    ) -> np.ndarray:
        """Compute exchange flow ratio (inflow / outflow)."""
        inflow = np.asarray(inflow, dtype=np.float64)
        outflow = np.asarray(outflow, dtype=np.float64)
        return inflow / (outflow + 1e-10)

    @staticmethod
    def normalize_onchain_features(
        df: pd.DataFrame, window: int = 100
    ) -> pd.DataFrame:
        """Apply rolling z-score normalization to all columns.

        Parameters
        ----------
        df : pd.DataFrame
            Raw on-chain features.
        window : int
            Rolling window for mean / std computation.

        Returns
        -------
        pd.DataFrame
            Z-score-normalized features.
        """
        rolling_mean = df.rolling(window=window, min_periods=1).mean()
        rolling_std = df.rolling(window=window, min_periods=1).std().fillna(0.0) + 1e-10
        return (df - rolling_mean) / rolling_std

from __future__ import annotations

"""Market microstructure feature computation module."""

import numpy as np
import pandas as pd


class MicrostructureFeatures:
    """Static methods for computing market microstructure features."""

    @staticmethod
    def compute_order_book_imbalance(
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
        levels: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute order-book imbalance at specified depth levels.

        Parameters
        ----------
        bid_volumes : np.ndarray
            2-D array of shape ``(n_ticks, n_levels)`` with bid sizes.
        ask_volumes : np.ndarray
            2-D array of shape ``(n_ticks, n_levels)`` with ask sizes.
        levels : list[int], optional
            Cumulative depth levels to evaluate (default ``[1, 5, 10]``).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping ``"obi_{level}"`` → 1-D imbalance array per level.
        """
        if levels is None:
            levels = [1, 5, 10]

        bid_volumes = np.asarray(bid_volumes, dtype=np.float64)
        ask_volumes = np.asarray(ask_volumes, dtype=np.float64)

        result: dict[str, np.ndarray] = {}
        for lvl in levels:
            cap = min(lvl, bid_volumes.shape[1] if bid_volumes.ndim > 1 else 1)
            if bid_volumes.ndim == 1:
                sum_bid = bid_volumes
                sum_ask = ask_volumes
            else:
                sum_bid = bid_volumes[:, :cap].sum(axis=1)
                sum_ask = ask_volumes[:, :cap].sum(axis=1)
            result[f"obi_{lvl}"] = (sum_bid - sum_ask) / (sum_bid + sum_ask + 1e-10)
        return result

    @staticmethod
    def compute_bid_ask_spread(
        best_bid: np.ndarray, best_ask: np.ndarray
    ) -> np.ndarray:
        """Compute bid-ask spread."""
        best_bid = np.asarray(best_bid, dtype=np.float64)
        best_ask = np.asarray(best_ask, dtype=np.float64)
        mid = (best_bid + best_ask) / 2.0
        return (best_ask - best_bid) / (mid + 1e-10)

    @staticmethod
    def compute_vpin(
        buy_volume: np.ndarray,
        sell_volume: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Compute Volume-Synchronized Probability of Informed Trading.

        VPIN = rolling mean of |buy_vol - sell_vol| / (buy_vol + sell_vol).
        """
        buy_volume = np.asarray(buy_volume, dtype=np.float64)
        sell_volume = np.asarray(sell_volume, dtype=np.float64)
        total = buy_volume + sell_volume + 1e-10
        abs_diff = np.abs(buy_volume - sell_volume)
        ratio = abs_diff / total
        result = np.full_like(ratio, np.nan)
        for i in range(window - 1, len(ratio)):
            result[i] = np.nanmean(ratio[i - window + 1 : i + 1])
        return result

    @staticmethod
    def compute_trade_flow_imbalance(
        trade_sides: np.ndarray,
        trade_volumes: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Compute rolling trade flow imbalance.

        Parameters
        ----------
        trade_sides : np.ndarray
            +1 for buy, -1 for sell.
        trade_volumes : np.ndarray
            Unsigned trade sizes.
        window : int
            Rolling window size.
        """
        trade_sides = np.asarray(trade_sides, dtype=np.float64)
        trade_volumes = np.asarray(trade_volumes, dtype=np.float64)
        signed = trade_sides * trade_volumes
        result = np.full(len(signed), np.nan)
        for i in range(window - 1, len(signed)):
            chunk = signed[i - window + 1 : i + 1]
            total = np.sum(np.abs(chunk)) + 1e-10
            result[i] = np.sum(chunk) / total
        return result

    @staticmethod
    def compute_cumulative_volume_delta(
        buy_volume: np.ndarray, sell_volume: np.ndarray
    ) -> np.ndarray:
        """Compute cumulative volume delta (CVD)."""
        buy_volume = np.asarray(buy_volume, dtype=np.float64)
        sell_volume = np.asarray(sell_volume, dtype=np.float64)
        return np.cumsum(buy_volume - sell_volume)

    @staticmethod
    def compute_kyle_lambda(
        price_changes: np.ndarray,
        signed_volume: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Compute Kyle's lambda via rolling OLS of price change on signed volume."""
        price_changes = np.asarray(price_changes, dtype=np.float64)
        signed_volume = np.asarray(signed_volume, dtype=np.float64)
        result = np.full(len(price_changes), np.nan)
        for i in range(window - 1, len(price_changes)):
            y = price_changes[i - window + 1 : i + 1]
            x = signed_volume[i - window + 1 : i + 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2) + 1e-10
            result[i] = np.sum((x - x_mean) * (y - y_mean)) / denom
        return result

    @staticmethod
    def compute_amihud_illiquidity(
        returns: np.ndarray,
        volume: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Compute Amihud illiquidity ratio (rolling)."""
        returns = np.asarray(returns, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        ratio = np.abs(returns) / (volume + 1e-10)
        result = np.full(len(ratio), np.nan)
        for i in range(window - 1, len(ratio)):
            result[i] = np.nanmean(ratio[i - window + 1 : i + 1])
        return result

    @staticmethod
    def compute_all(
        order_book_df: pd.DataFrame, trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute all microstructure features from order-book and trade data.

        Parameters
        ----------
        order_book_df : pd.DataFrame
            Expected columns: best_bid, best_ask and optionally
            bid_vol_1..bid_vol_10, ask_vol_1..ask_vol_10.
        trades_df : pd.DataFrame
            Expected columns: side (+1/-1), volume, price.

        Returns
        -------
        pd.DataFrame
        """
        m = MicrostructureFeatures
        result = pd.DataFrame(index=trades_df.index)

        # Bid-ask spread
        if {"best_bid", "best_ask"}.issubset(order_book_df.columns):
            result["bid_ask_spread"] = m.compute_bid_ask_spread(
                order_book_df["best_bid"].values, order_book_df["best_ask"].values
            )

        # Order book imbalance (if level columns present)
        bid_cols = sorted([c for c in order_book_df.columns if c.startswith("bid_vol_")])
        ask_cols = sorted([c for c in order_book_df.columns if c.startswith("ask_vol_")])
        if bid_cols and ask_cols:
            obi = m.compute_order_book_imbalance(
                order_book_df[bid_cols].values,
                order_book_df[ask_cols].values,
            )
            for key, arr in obi.items():
                result[key] = arr

        # Trade-based features
        if {"side", "volume"}.issubset(trades_df.columns):
            sides = trades_df["side"].values
            vols = trades_df["volume"].values
            buy_vol = np.where(sides > 0, vols, 0.0)
            sell_vol = np.where(sides < 0, vols, 0.0)
            result["vpin"] = m.compute_vpin(buy_vol, sell_vol)
            result["tfi"] = m.compute_trade_flow_imbalance(sides, vols)
            result["cvd"] = m.compute_cumulative_volume_delta(buy_vol, sell_vol)

        if {"price", "side", "volume"}.issubset(trades_df.columns):
            price_changes = np.diff(trades_df["price"].values, prepend=np.nan)
            signed_vol = trades_df["side"].values * trades_df["volume"].values
            result["kyle_lambda"] = m.compute_kyle_lambda(price_changes, signed_vol)

            returns = price_changes / (trades_df["price"].values + 1e-10)
            result["amihud"] = m.compute_amihud_illiquidity(returns, vols)

        return result

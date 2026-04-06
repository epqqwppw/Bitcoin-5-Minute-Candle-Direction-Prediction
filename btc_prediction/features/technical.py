from __future__ import annotations

"""Technical analysis feature computation module."""

import numpy as np
import pandas as pd


class TechnicalFeatures:
    """Collection of static methods for computing technical indicators."""

    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        period : int
            Look-back period (default 14).

        Returns
        -------
        pd.Series
            RSI values in [0, 100].
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return pd.Series(100.0 - 100.0 / (1.0 + rs), index=close.index, name="rsi")

    @staticmethod
    def compute_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Compute MACD, signal line and histogram.

        Returns DataFrame with columns: macd, signal, histogram.
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram},
            index=close.index,
        )

    @staticmethod
    def compute_bollinger_bands(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands.

        Returns DataFrame with columns: upper, middle, lower, bandwidth.
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        bandwidth = (upper - lower) / (middle + 1e-10)
        return pd.DataFrame(
            {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth},
            index=close.index,
        )

    @staticmethod
    def compute_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Compute Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return pd.Series(
            tr.ewm(span=period, adjust=False).mean(), index=close.index, name="atr"
        )

    @staticmethod
    def compute_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        return pd.Series(cum_tp_vol / (cum_vol + 1e-10), index=close.index, name="vwap")

    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0.0)
        return pd.Series((direction * volume).cumsum(), index=close.index, name="obv")

    @staticmethod
    def compute_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Compute Stochastic Oscillator (%K and %D)."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return pd.DataFrame({"k": k, "d": d}, index=close.index)

    @staticmethod
    def compute_cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Compute Commodity Channel Index."""
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return pd.Series(
            (tp - sma_tp) / (0.015 * mad + 1e-10), index=close.index, name="cci"
        )

    @staticmethod
    def compute_williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Compute Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return pd.Series(wr, index=close.index, name="williams_r")

    @staticmethod
    def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
        """Compute Rate of Change."""
        prev = close.shift(period)
        return pd.Series(
            (close - prev) / (prev + 1e-10) * 100.0, index=close.index, name="roc"
        )

    @staticmethod
    def compute_ema(series: pd.Series, period: int) -> pd.Series:
        """Compute Exponential Moving Average."""
        return pd.Series(
            series.ewm(span=period, adjust=False).mean(),
            index=series.index,
            name=f"ema_{period}",
        )

    @staticmethod
    def compute_sma(series: pd.Series, period: int) -> pd.Series:
        """Compute Simple Moving Average."""
        return pd.Series(
            series.rolling(window=period).mean(),
            index=series.index,
            name=f"sma_{period}",
        )

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators for a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            Original columns plus all computed indicators.
        """
        result = df.copy()
        t = TechnicalFeatures
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI at multiple periods
        for p in (7, 14, 21):
            result[f"rsi_{p}"] = t.compute_rsi(close, period=p)

        # MACD
        macd_df = t.compute_macd(close)
        result["macd"] = macd_df["macd"]
        result["macd_signal"] = macd_df["signal"]
        result["macd_histogram"] = macd_df["histogram"]

        # Bollinger Bands at multiple periods
        for p in (20,):
            bb = t.compute_bollinger_bands(close, period=p)
            for col in bb.columns:
                result[f"bb_{p}_{col}"] = bb[col]

        # ATR
        for p in (7, 14):
            result[f"atr_{p}"] = t.compute_atr(high, low, close, period=p)

        # VWAP
        result["vwap"] = t.compute_vwap(high, low, close, volume)

        # OBV
        result["obv"] = t.compute_obv(close, volume)

        # Stochastic
        stoch = t.compute_stochastic(high, low, close)
        result["stoch_k"] = stoch["k"]
        result["stoch_d"] = stoch["d"]

        # CCI
        result["cci"] = t.compute_cci(high, low, close)

        # Williams %R
        result["williams_r"] = t.compute_williams_r(high, low, close)

        # ROC
        for p in (6, 12):
            result[f"roc_{p}"] = t.compute_roc(close, period=p)

        # Moving averages
        for p in (5, 10, 20, 50):
            result[f"ema_{p}"] = t.compute_ema(close, p)
            result[f"sma_{p}"] = t.compute_sma(close, p)

        # Handle NaN: forward fill then backward fill
        result = result.ffill().bfill()
        return result

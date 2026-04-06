from __future__ import annotations

"""Advanced / complexity-based feature computation module."""

from math import factorial

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


class AdvancedFeatures:
    """Static methods for advanced statistical and complexity features."""

    @staticmethod
    def compute_shannon_entropy(
        series: np.ndarray, bins: int = 50, window: int = 50
    ) -> np.ndarray:
        """Compute rolling Shannon entropy.

        Parameters
        ----------
        series : np.ndarray
            1-D numeric array.
        bins : int
            Number of histogram bins.
        window : int
            Rolling window size.
        """
        series = np.asarray(series, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            chunk = series[i - window + 1 : i + 1]
            counts, _ = np.histogram(chunk, bins=bins)
            probs = counts / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            result[i] = -np.sum(probs * np.log2(probs + 1e-10))
        return result

    @staticmethod
    def compute_permutation_entropy(
        series: np.ndarray,
        order: int = 3,
        delay: int = 1,
        window: int = 50,
    ) -> np.ndarray:
        """Compute rolling permutation entropy.

        Parameters
        ----------
        series : np.ndarray
            1-D numeric array.
        order : int
            Embedding dimension.
        delay : int
            Time delay.
        window : int
            Rolling window size.
        """
        series = np.asarray(series, dtype=np.float64)
        result = np.full(len(series), np.nan)
        max_pe = np.log2(factorial(order))

        for i in range(window - 1, len(series)):
            chunk = series[i - window + 1 : i + 1]
            n = len(chunk)
            n_patterns = n - (order - 1) * delay
            if n_patterns <= 0:
                continue
            patterns: dict[tuple[int, ...], int] = {}
            for j in range(n_patterns):
                indices = [j + k * delay for k in range(order)]
                motif = chunk[indices]
                perm = tuple(np.argsort(motif).tolist())
                patterns[perm] = patterns.get(perm, 0) + 1
            counts = np.array(list(patterns.values()), dtype=np.float64)
            probs = counts / counts.sum()
            pe = -np.sum(probs * np.log2(probs + 1e-10))
            result[i] = pe / (max_pe + 1e-10)
        return result

    @staticmethod
    def compute_hurst_exponent(
        series: np.ndarray, window: int = 100
    ) -> np.ndarray:
        """Compute rolling Hurst exponent via rescaled range (R/S) analysis."""
        series = np.asarray(series, dtype=np.float64)
        result = np.full(len(series), np.nan)

        for i in range(window - 1, len(series)):
            chunk = series[i - window + 1 : i + 1]
            mean_val = np.mean(chunk)
            deviations = chunk - mean_val
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)
            s = np.std(chunk, ddof=1) + 1e-10
            rs = r / s
            if rs > 0:
                result[i] = np.log(rs + 1e-10) / (np.log(window) + 1e-10)
            else:
                result[i] = 0.5
        return result

    @staticmethod
    def compute_fractal_dimension(
        series: np.ndarray, window: int = 100
    ) -> np.ndarray:
        """Compute rolling fractal dimension using box-counting approximation.

        Uses Higuchi's simplified approach: FD ≈ 2 - H (from Hurst exponent).
        """
        hurst = AdvancedFeatures.compute_hurst_exponent(series, window=window)
        return 2.0 - hurst

    @staticmethod
    def compute_sample_entropy(
        series: np.ndarray,
        m: int = 2,
        r: float = 0.2,
        window: int = 100,
    ) -> np.ndarray:
        """Compute rolling sample entropy.

        Parameters
        ----------
        series : np.ndarray
            1-D numeric array.
        m : int
            Embedding dimension.
        r : float
            Tolerance (fraction of std).
        window : int
            Rolling window size.
        """
        series = np.asarray(series, dtype=np.float64)
        result = np.full(len(series), np.nan)

        def _count_matches(data: np.ndarray, dim: int, tol: float) -> int:
            n = len(data)
            count = 0
            for a in range(n - dim):
                for b in range(a + 1, n - dim):
                    if np.max(np.abs(data[a : a + dim] - data[b : b + dim])) < tol:
                        count += 1
            return count

        for i in range(window - 1, len(series)):
            chunk = series[i - window + 1 : i + 1]
            std = np.std(chunk) + 1e-10
            tol = r * std
            a = _count_matches(chunk, m, tol)
            b = _count_matches(chunk, m + 1, tol)
            if a == 0:
                result[i] = 0.0
            else:
                result[i] = -np.log((b + 1e-10) / (a + 1e-10))
        return result

    @staticmethod
    def compute_wavelet_features(
        series: np.ndarray,
        wavelet: str = "db4",
        level: int = 4,
    ) -> np.ndarray:
        """Extract wavelet-like features using bandpass filter decomposition.

        Uses ``scipy.signal`` Butterworth filters to create a multi-resolution
        decomposition analogous to DWT, computing energy and entropy per band.

        Parameters
        ----------
        series : np.ndarray
            1-D numeric array.
        wavelet : str
            Placeholder for wavelet name (unused; kept for API compat).
        level : int
            Number of decomposition levels.

        Returns
        -------
        np.ndarray
            Array of shape ``(2 * level,)`` with [energy_1, ..., energy_L,
            entropy_1, ..., entropy_L].
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 16:
            return np.zeros(2 * level)

        energies: list[float] = []
        entropies: list[float] = []
        nyquist = 0.5

        for lv in range(1, level + 1):
            low = nyquist / (2 ** (lv + 1))
            high = nyquist / (2**lv)
            # Clamp to valid Butterworth range (0, 1) exclusive
            low_n = max(low / nyquist, 0.01)
            high_n = min(high / nyquist, 0.99)
            if low_n >= high_n:
                energies.append(0.0)
                entropies.append(0.0)
                continue
            try:
                sos = scipy_signal.butter(3, [low_n, high_n], btype="band", output="sos")
                coeff = scipy_signal.sosfiltfilt(sos, series)
            except (ValueError, np.linalg.LinAlgError):
                energies.append(0.0)
                entropies.append(0.0)
                continue

            energy = float(np.sum(coeff**2))
            energies.append(energy)

            # Entropy of squared coefficients
            sq = coeff**2
            total = sq.sum() + 1e-10
            probs = sq / total
            probs = probs[probs > 0]
            ent = float(-np.sum(probs * np.log2(probs + 1e-10)))
            entropies.append(ent)

        return np.array(energies + entropies)

    @staticmethod
    def compute_all(close: np.ndarray) -> pd.DataFrame:
        """Compute all advanced features from a close-price array.

        Parameters
        ----------
        close : np.ndarray
            1-D array of closing prices.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each advanced feature.
        """
        a = AdvancedFeatures
        close = np.asarray(close, dtype=np.float64)
        result: dict[str, np.ndarray] = {}
        result["shannon_entropy"] = a.compute_shannon_entropy(close)
        result["permutation_entropy"] = a.compute_permutation_entropy(close)
        result["hurst_exponent"] = a.compute_hurst_exponent(close)
        result["fractal_dimension"] = a.compute_fractal_dimension(close)
        result["sample_entropy"] = a.compute_sample_entropy(close)

        wf = a.compute_wavelet_features(close)
        for i, val in enumerate(wf):
            result[f"wavelet_{i}"] = np.full(len(close), val)

        return pd.DataFrame(result)

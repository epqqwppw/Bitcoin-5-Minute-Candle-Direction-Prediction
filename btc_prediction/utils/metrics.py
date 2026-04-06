"""Evaluation metrics for classification, calibration, and trading performance.

All functions accept numpy arrays and return plain Python floats.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef as sklearn_mcc,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def accuracy(y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
) -> dict[str, float]:
    """Compute precision, recall, and F1-score.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dictionary with keys ``precision``, ``recall``, and ``f1``.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
    }


def matthews_corrcoef(y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]) -> float:
    """Compute Matthew's correlation coefficient.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        MCC value in [-1, 1].
    """
    return float(sklearn_mcc(y_true, y_pred))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def brier_score(
    probabilities: npt.NDArray[np.floating],
    actuals: npt.NDArray[np.int_],
) -> float:
    """Compute the Brier score (mean squared error of probability estimates).

    Args:
        probabilities: Predicted probabilities in [0, 1].
        actuals: Binary ground-truth labels.

    Returns:
        Brier score (lower is better).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    return float(np.mean((probabilities - actuals) ** 2))


def expected_calibration_error(
    probabilities: npt.NDArray[np.floating],
    actuals: npt.NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` equal-width bins by predicted
    probability and measures the weighted average gap between predicted
    confidence and observed accuracy.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        actuals: Binary ground-truth labels.
        n_bins: Number of calibration bins.

    Returns:
        ECE value (lower is better).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total = len(probabilities)
    if total == 0:
        return 0.0

    for idx, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (probabilities >= lo if idx == 0 else probabilities > lo) & (probabilities <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(probabilities[mask].mean())
        avg_accuracy = float(actuals[mask].mean())
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return ece


# ---------------------------------------------------------------------------
# Trading / portfolio metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: npt.NDArray[np.floating],
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualised Sharpe ratio.

    Assumes returns are per-period (e.g. per 5-minute bar).  Annualisation
    uses 252 trading days × 288 five-minute bars per day.

    Args:
        returns: Array of per-period returns.
        risk_free_rate: Per-period risk-free rate.

    Returns:
        Annualised Sharpe ratio, or 0.0 when volatility is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    if std == 0.0:
        return 0.0
    periods_per_year = 252 * 288  # 5-min bars in a trading year
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: npt.NDArray[np.floating],
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualised Sortino ratio.

    Uses downside deviation instead of total standard deviation.

    Args:
        returns: Array of per-period returns.
        risk_free_rate: Per-period risk-free rate.

    Returns:
        Annualised Sortino ratio, or 0.0 when downside deviation is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    if downside_std == 0.0:
        return 0.0
    periods_per_year = 252 * 288
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: npt.NDArray[np.floating]) -> float:
    """Compute the maximum drawdown from an equity curve.

    Args:
        equity_curve: Cumulative equity values over time.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.15 means 15 %).
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if len(equity_curve) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.where(running_max == 0, 1.0, running_max)
    return float(np.max(drawdowns))


def profit_factor(returns: npt.NDArray[np.floating]) -> float:
    """Compute the profit factor (gross profit / gross loss).

    Args:
        returns: Array of per-period returns.

    Returns:
        Profit factor.  Returns ``float('inf')`` when there are no losses.
    """
    returns = np.asarray(returns, dtype=np.float64)
    gross_profit = float(np.sum(returns[returns > 0]))
    gross_loss = float(np.abs(np.sum(returns[returns < 0])))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calmar_ratio(
    returns: npt.NDArray[np.floating],
    equity_curve: npt.NDArray[np.floating],
) -> float:
    """Compute the Calmar ratio (annualised return / max drawdown).

    Args:
        returns: Array of per-period returns.
        equity_curve: Cumulative equity values over time.

    Returns:
        Calmar ratio, or 0.0 when max drawdown is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    periods_per_year = 252 * 288
    annualised_return = float(np.mean(returns)) * periods_per_year
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return 0.0
    return annualised_return / mdd


def win_rate(returns: npt.NDArray[np.floating]) -> float:
    """Compute the win rate (fraction of positive-return periods).

    Args:
        returns: Array of per-period returns.

    Returns:
        Win rate in [0, 1].
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))

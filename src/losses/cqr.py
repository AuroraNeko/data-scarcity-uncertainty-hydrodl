"""
CQR: Conformalized Quantile Regression for uncertainty calibration.

Paper Section 11.3-11.4:
    1. Compute nonconformity scores on calibration (validation) set
    2. Adjust prediction intervals by q_cal to achieve target coverage

Algorithm:
    s_i = max(Q_lower_i - y_i, y_i - Q_upper_i)
    q_cal = quantile(s, 1 - alpha)
    adjusted_interval = [Q_lower - q_cal, Q_upper + q_cal]

Provides distribution-free coverage guarantee: P(Y in [Q_lower - q_cal, Q_upper + q_cal]) >= 1 - alpha
"""

import numpy as np
import torch
from torch import Tensor


class CQRCalibrator:
    """Conformalized Quantile Regression post-hoc calibrator."""

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: significance level (0.1 for 90% intervals)
        """
        self.alpha = alpha
        self.q_cal = None

    def fit(self, q_lower: np.ndarray, q_upper: np.ndarray, y_obs: np.ndarray) -> float:
        """Compute calibration quantile from validation set.

        Args:
            q_lower: predicted lower quantile (e.g. Q_0.05)
            q_upper: predicted upper quantile (e.g. Q_0.95)
            y_obs: observed values
        Returns:
            q_cal: calibration adjustment
        """
        # Nonconformity scores
        scores = np.maximum(q_lower - y_obs, y_obs - q_upper)

        # Calibration quantile with finite-sample correction
        n = len(scores)
        q_cal = np.quantile(scores, min((1 - self.alpha) * (1 + 1 / n), 1.0))

        self.q_cal = q_cal
        return q_cal

    def calibrate(self, q_lower: np.ndarray, q_upper: np.ndarray):
        """Adjust prediction intervals.

        Returns:
            (adjusted_lower, adjusted_upper)
        """
        assert self.q_cal is not None, "Must call fit() first"
        return q_lower - self.q_cal, q_upper + self.q_cal


def compute_uncertainty_metrics(
    q_lower: np.ndarray, q_upper: np.ndarray, y_obs: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """Compute uncertainty evaluation metrics.

    Returns:
        dict with PICP, MPIW, Winkler score, and coverage by flow regime
    """
    # PICP: Prediction Interval Coverage Probability
    in_interval = (y_obs >= q_lower) & (y_obs <= q_upper)
    picp = in_interval.mean()

    # MPIW: Mean Prediction Interval Width
    mpiw = (q_upper - q_lower).mean()

    # Winkler Score (penalizes both missing coverage and wide intervals)
    width = q_upper - q_lower
    penalty_lower = (2 / alpha) * np.maximum(q_lower - y_obs, 0)
    penalty_upper = (2 / alpha) * np.maximum(y_obs - q_upper, 0)
    winkler = (width + penalty_lower + penalty_upper).mean()

    # Coverage by flow regime
    q33, q67 = np.percentile(y_obs, [33, 67])
    low = y_obs <= q33
    normal = (y_obs > q33) & (y_obs <= q67)
    high = y_obs > q67

    return {
        "picp": float(picp),
        "target_coverage": float(1 - alpha),
        "mpiw": float(mpiw),
        "winkler_score": float(winkler),
        "coverage_low_flow": float(in_interval[low].mean()) if low.sum() > 0 else 0.0,
        "coverage_normal_flow": float(in_interval[normal].mean()) if normal.sum() > 0 else 0.0,
        "coverage_high_flow": float(in_interval[high].mean()) if high.sum() > 0 else 0.0,
    }

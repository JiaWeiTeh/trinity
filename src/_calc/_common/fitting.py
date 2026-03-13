# -*- coding: utf-8 -*-
"""
Reusable OLS fitting with iterative sigma-clipping.

Every analysis script in ``src._calc`` performs the same OLS + sigma-clip
procedure.  This module provides the single canonical implementation.
"""

from typing import Dict, Optional

import numpy as np


def ols_sigma_clip(
    X: np.ndarray,
    y: np.ndarray,
    sigma_clip: float,
    max_iter: int = 10,
) -> Optional[Dict]:
    """Ordinary least-squares with iterative sigma-clipping.

    Parameters
    ----------
    X : (N, k) design matrix (including intercept column if desired).
    y : (N,) response vector.
    sigma_clip : reject points whose |residual| > sigma_clip * rms.
    max_iter : maximum clipping iterations.

    Returns
    -------
    dict with keys ``beta``, ``unc``, ``R2``, ``rms_dex``, ``n_used``,
    ``n_rejected``, ``mask``, ``y_pred``, or *None* if the system is
    under-determined.
    """
    n_total = len(y)
    mask = np.ones(n_total, dtype=bool)

    for _ in range(max_iter):
        X_use, y_use = X[mask], y[mask]
        n_use = mask.sum()
        if n_use < X.shape[1]:
            return None
        XtX = X_use.T @ X_use
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return None
        beta = XtX_inv @ (X_use.T @ y_use)

        resid = y - X @ beta
        rms = np.std(resid[mask], ddof=X.shape[1])
        if rms == 0:
            break
        new_mask = np.abs(resid) <= sigma_clip * rms
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    n_used = int(mask.sum())
    y_pred = X @ beta
    ss_res = np.sum((y[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms_dex = np.sqrt(ss_res / max(n_used - X.shape[1], 1))
    s2 = ss_res / max(n_used - X.shape[1], 1)
    unc = np.sqrt(np.diag(s2 * XtX_inv))

    return {
        "beta": beta,
        "unc": unc,
        "R2": R2,
        "rms_dex": rms_dex,
        "n_used": n_used,
        "n_rejected": n_total - n_used,
        "mask": mask,
        "y_pred": y_pred,
    }


def logsumexp(log_vals: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(log_vals)
    if not np.isfinite(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(log_vals - a_max)))

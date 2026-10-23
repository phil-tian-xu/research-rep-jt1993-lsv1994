"""OLS with Newey-West inference for monthly factor regressions."""

from math import erfc, sqrt
from statistics import NormalDist

import numpy as np


def ols_newey_west(y, x, lags=5):
    """Add an intercept and estimate OLS with Bartlett Newey-West covariance.

    y is one-dimensional; x contains only the regressors, without a constant.
    Observations must be finite, consecutive, and ordered in time; callers
    must validate calendar coverage before passing arrays. The covariance uses
    the n/(n-p) correction, where p includes the intercept. Reported t-statistics
    are coefficient/SE; two-sided p-values and 95% intervals use the asymptotic
    normal distribution, not a finite-sample Student t approximation.
    With lags=0 the covariance is HC1, not homoskedastic OLS covariance.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("Require a one-dimensional y and a two-dimensional x with matching rows.")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("OLS requires finite observations; missing months must not be silently dropped.")
    n, p = len(y), x.shape[1] + 1
    if n <= p:
        raise ValueError("Need more observations than coefficients, including the intercept.")
    if not isinstance(lags, int) or isinstance(lags, bool) or not 0 <= lags < n:
        raise ValueError("Newey-West lags must be an integer >=0 and smaller than the sample size.")
    design = np.column_stack([np.ones(n), x])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != p:
        raise ValueError("The regression design is rank deficient; check constant or collinear factors.")
    fitted = design @ beta
    residual = y - fitted
    total_sum_squares = float((y - y.mean()) @ (y - y.mean()))
    if total_sum_squares <= 0:
        raise ValueError("The dependent variable must vary over time.")
    score = design * residual[:, None]
    meat = score.T @ score
    for lag in range(1, lags + 1):
        cross = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    covariance = (bread @ meat @ bread) * n / (n - p)
    covariance = (covariance + covariance.T) / 2
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    if not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("Degenerate Newey-West standard errors; check the regression inputs.")
    t_stat = beta / se
    critical = NormalDist().inv_cdf(0.975)
    r_squared = 1 - float(residual @ residual) / total_sum_squares
    return {
        "coefficient": beta, "hac_se": se, "t_stat": t_stat,
        "p_value": np.array([erfc(abs(value) / sqrt(2)) for value in t_stat]),
        "ci95_lower": beta - critical * se, "ci95_upper": beta + critical * se,
        "covariance": covariance, "fitted": fitted, "residual": residual,
        "n_obs": n, "df_resid": n - p, "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (n - 1) / (n - p),
    }

"""Realized volatility estimators for historical price data."""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def close_to_close(prices: pd.Series, window: int = 21) -> pd.Series:
    """Standard close-to-close realized vol (annualized)."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def parkinson(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson high-low range estimator (annualized)."""
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * log_hl_sq.rolling(window=window).mean()
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def yang_zhang(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Yang-Zhang OHLC estimator (annualized, drift-independent)."""
    log_co = np.log(open / close.shift(1))
    log_oc = np.log(close / open)
    log_ho = np.log(high / open)
    log_lo = np.log(low / open)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)

    overnight_var = log_co.rolling(window=window).var()
    open_close_var = log_oc.rolling(window=window).var()

    rs_daily = log_ho * log_hc + log_lo * log_lc
    rs_var = rs_daily.rolling(window=window).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    yz_var = overnight_var + k * open_close_var + (1.0 - k) * rs_var

    return np.sqrt(np.maximum(yz_var, 0.0) * TRADING_DAYS_PER_YEAR)


def variance_risk_premium(
    realized_vol: float | pd.Series,
    implied_vol_atm: float | pd.Series,
) -> float | pd.Series:
    """VRP = implied vol - realized vol."""
    return implied_vol_atm - realized_vol

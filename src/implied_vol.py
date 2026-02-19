"""Implied volatility solver using Newton-Raphson with bisection fallback."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from src import config
from src.pricing import BlackScholes


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver with bisection fallback."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    option_type = option_type.lower().strip()
    if option_type not in ("call", "put"):
        return np.nan

    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < intrinsic:
        return np.nan

    # Brenner-Subrahmanyam initial guess
    sigma = np.sqrt(2.0 * np.pi / T) * market_price / S
    sigma = np.clip(sigma, 0.01, 3.0)

    for _ in range(max_iter):
        bs = BlackScholes(S, K, T, r, q, sigma)
        price = bs.price(option_type)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega_raw = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        if abs(vega_raw) < 1e-12:
            return _bisection_iv(market_price, S, K, T, r, q, option_type)

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega_raw
        sigma = np.clip(sigma, 0.001, 5.0)

    return np.nan


def _bisection_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    low: float = 0.001,
    high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Bisection fallback for IV computation."""
    option_type = option_type.lower().strip()

    bs_low = BlackScholes(S, K, T, r, q, low)
    bs_high = BlackScholes(S, K, T, r, q, high)
    price_low = bs_low.price(option_type)
    price_high = bs_high.price(option_type)

    if (price_low - market_price) * (price_high - market_price) > 0:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        bs_mid = BlackScholes(S, K, T, r, q, mid)
        price_mid = bs_mid.price(option_type)
        diff = price_mid - market_price

        if abs(diff) < tol:
            return mid

        if diff > 0:
            high = mid
        else:
            low = mid

        if (high - low) < tol * 0.01:
            return mid

    return np.nan


def compute_all_ivs(
    data: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    """Compute IVs for an entire options DataFrame."""
    df = data.copy()
    total = len(df)

    computed_ivs = np.full(total, np.nan)

    for i, row in tqdm(
        df.iterrows(), total=total, desc="Computing IVs", unit="opt"
    ):
        iv = implied_vol(
            market_price=row["mid_price"],
            S=spot,
            K=row["strike"],
            T=row["T"],
            r=r,
            q=q,
            option_type=row["option_type"],
        )
        computed_ivs[df.index.get_loc(i)] = iv

    df["computed_iv"] = computed_ivs
    df["iv_converged"] = ~np.isnan(df["computed_iv"])

    converged = df["iv_converged"].sum()

    mask = (
        df["iv_converged"]
        & (df["computed_iv"] >= config.IV_LOWER_BOUND)
        & (df["computed_iv"] <= config.IV_UPPER_BOUND)
    )
    filtered_count = converged - mask.sum()
    df = df[mask].copy()

    print(
        f"Computed IVs: {converged}/{total} converged, "
        f"{filtered_count} filtered"
    )

    return df

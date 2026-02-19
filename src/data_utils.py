"""Data utilities for options volatility surface construction."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_risk_free_rate() -> float:
    """Fetch 3-month Treasury rate from FRED, fallback to config default."""
    try:
        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable not set")

        fred = Fred(api_key=api_key)
        series = fred.get_series("DGS3MO").dropna()
        rate = float(series.iloc[-1]) / 100.0
        logger.info("Risk-free rate from FRED (DGS3MO): %.4f", rate)
        return rate
    except Exception as exc:
        logger.warning(
            "Could not fetch risk-free rate from FRED (%s). "
            "Using fallback: %.4f",
            exc,
            config.FALLBACK_RISK_FREE_RATE,
        )
        return config.FALLBACK_RISK_FREE_RATE


def fetch_dividend_yield(ticker: str = config.DEFAULT_TICKER) -> float:
    """Fetch trailing dividend yield via yfinance."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        div_yield = info.get("dividendYield") or info.get(
            "trailingAnnualDividendYield"
        )
        if div_yield is None or div_yield == 0:
            raise ValueError("Dividend yield not available in ticker info")
        div_yield = float(div_yield)
        logger.info("Dividend yield for %s from yfinance: %.4f", ticker, div_yield)
        return div_yield
    except Exception as exc:
        logger.warning(
            "Could not fetch dividend yield for %s (%s). Using fallback: %.4f",
            ticker,
            exc,
            config.FALLBACK_DIVIDEND_YIELD,
        )
        return config.FALLBACK_DIVIDEND_YIELD


def fetch_options_chain(
    ticker: str = config.DEFAULT_TICKER,
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Fetch and clean full options chain for a ticker."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}")
    spot = float(hist["Close"].iloc[-1])
    logger.info("Spot price for %s: %.2f", ticker, spot)

    r = fetch_risk_free_rate()
    q = fetch_dividend_yield(ticker)

    now = datetime.now()
    frames = []

    for exp_str in stock.options:
        try:
            chain = stock.option_chain(exp_str)
        except Exception as exc:
            logger.warning("Skipping expiration %s: %s", exp_str, exc)
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            if df.empty:
                continue
            df = df.copy()
            exp_dt = pd.Timestamp(exp_str)
            df["expiration"] = exp_dt
            df["option_type"] = opt_type
            df["spot"] = spot
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
            bid_ask_denom = df["mid_price"].replace(0, np.nan)
            df["bid_ask_spread_pct"] = (df["ask"] - df["bid"]) / bid_ask_denom
            df["moneyness"] = df["strike"] / spot
            df["T"] = (exp_dt - pd.Timestamp(now)).days / 365.0
            df["intrinsic_value"] = np.where(
                df["option_type"] == "call",
                np.maximum(spot - df["strike"], 0),
                np.maximum(df["strike"] - spot, 0),
            )
            df["time_value"] = df["mid_price"] - df["intrinsic_value"]
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No options data retrieved for {ticker}")

    raw_df = pd.concat(frames, ignore_index=True)
    logger.info("Fetched %d raw option rows for %s", len(raw_df), ticker)

    cleaned_df = clean_options_data(raw_df, spot)

    if save_raw:
        date_tag = now.strftime("%Y%m%d")
        path = RAW_DATA_DIR / f"{ticker}_options_{date_tag}.parquet"
        cleaned_df.to_parquet(path, index=False)
        logger.info("Saved options data to %s", path)

    return cleaned_df, spot, r, q


def clean_options_data(data: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Apply sequential quality filters to raw options data."""
    orig = len(data)
    df = data.copy()

    def _log_step(name: str, before: int, after: int) -> None:
        removed = before - after
        if removed:
            logger.info("  Filter [%s]: removed %d rows", name, removed)

    n_before = len(df)
    df = df[df["bid"] > 0]
    _log_step("bid==0", n_before, len(df))

    n_before = len(df)
    vol = df["volume"].fillna(0)
    oi = df["openInterest"].fillna(0)
    df = df[(vol + oi) >= config.MIN_VOLUME_PLUS_OI]
    _log_step("volume+OI < MIN", n_before, len(df))

    n_before = len(df)
    df = df[df["bid_ask_spread_pct"] <= config.MAX_BID_ASK_SPREAD_PCT]
    _log_step("spread > MAX", n_before, len(df))

    n_before = len(df)
    df = df[
        (df["moneyness"] >= config.MONEYNESS_LOWER)
        & (df["moneyness"] <= config.MONEYNESS_UPPER)
    ]
    _log_step("moneyness out of bounds", n_before, len(df))

    n_before = len(df)
    df = df[df["mid_price"] >= config.MIN_OPTION_PRICE]
    _log_step("mid_price < MIN", n_before, len(df))

    n_before = len(df)
    df = df[df["T"] >= config.MIN_DAYS_TO_EXPIRY / 365.0]
    _log_step("T < MIN_DAYS", n_before, len(df))

    n_before = len(df)
    df = df[df["time_value"] >= 0]
    _log_step("time_value < 0", n_before, len(df))

    final = len(df)
    removed = orig - final
    print(f"Cleaned: {orig} -> {final} rows ({removed} removed)")
    logger.info("Cleaned: %d -> %d rows (%d removed)", orig, final, removed)

    return df.reset_index(drop=True)


def load_cached_data(
    ticker: str = config.DEFAULT_TICKER,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Load cached options data or fetch fresh if stale."""
    pattern = f"{ticker}_options_*.parquet"
    candidates = sorted(RAW_DATA_DIR.glob(pattern), reverse=True)

    if candidates:
        latest = candidates[0]
        age = datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)
        if age < timedelta(hours=24):
            logger.info("Loading cached data from %s (age: %s)", latest, age)
            df = pd.read_parquet(latest)
            spot = float(df["spot"].iloc[0])
            r = fetch_risk_free_rate()
            q = fetch_dividend_yield(ticker)
            return df, spot, r, q
        else:
            logger.info("Cache is stale (%s old). Fetching fresh data.", age)

    return fetch_options_chain(ticker, save_raw=True)


def compute_forward_from_put_call_parity(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    r: float,
    T: float,
) -> float:
    """Estimate forward price from put-call parity."""
    try:
        calls = calls_df[["strike", "mid_price"]].rename(
            columns={"mid_price": "call_price"}
        )
        puts = puts_df[["strike", "mid_price"]].rename(
            columns={"mid_price": "put_price"}
        )
        merged = pd.merge(calls, puts, on="strike", how="inner")

        if merged.empty:
            raise ValueError("No matched strikes for put-call parity")

        # F = K + e^(rT) * (C - P)
        forwards = merged["strike"] + np.exp(r * T) * (
            merged["call_price"] - merged["put_price"]
        )
        forward = float(forwards.median())
        logger.info(
            "Forward from put-call parity (T=%.3f): %.2f "
            "(from %d matched strikes)",
            T,
            forward,
            len(merged),
        )
        return forward
    except Exception as exc:
        spot = float(
            calls_df["spot"].iloc[0]
            if "spot" in calls_df.columns
            else puts_df["spot"].iloc[0]
        )
        q = fetch_dividend_yield()
        forward = spot * np.exp((r - q) * T)
        logger.warning(
            "Put-call parity failed (%s). Fallback forward: %.2f", exc, forward
        )
        return forward


def build_otm_smile(data: pd.DataFrame, forward: float) -> pd.DataFrame:
    """Build OTM implied vol smile from options data."""
    atm_tol = 0.005

    calls = data[data["option_type"] == "call"].copy()
    puts = data[data["option_type"] == "put"].copy()

    records = []

    all_strikes = sorted(data["strike"].unique())
    for K in all_strikes:
        ratio = K / forward
        near_atm = abs(ratio - 1.0) <= atm_tol

        c_row = calls[calls["strike"] == K]
        p_row = puts[puts["strike"] == K]

        if near_atm and not c_row.empty and not p_row.empty:
            iv = (
                float(c_row["impliedVolatility"].iloc[0])
                + float(p_row["impliedVolatility"].iloc[0])
            ) / 2.0
            opt_used = "average"
        elif K < forward and not p_row.empty:
            iv = float(p_row["impliedVolatility"].iloc[0])
            opt_used = "put"
        elif K > forward and not c_row.empty:
            iv = float(c_row["impliedVolatility"].iloc[0])
            opt_used = "call"
        elif not c_row.empty:
            iv = float(c_row["impliedVolatility"].iloc[0])
            opt_used = "call"
        elif not p_row.empty:
            iv = float(p_row["impliedVolatility"].iloc[0])
            opt_used = "put"
        else:
            continue

        records.append(
            {
                "strike": K,
                "IV": iv,
                "option_type_used": opt_used,
                "moneyness": K / forward,
                "log_moneyness": float(np.log(K / forward)),
            }
        )

    smile = pd.DataFrame(records)
    logger.info(
        "Built OTM smile with %d strikes (forward=%.2f)", len(smile), forward
    )
    return smile


def fetch_underlying_history(
    ticker: str = config.DEFAULT_TICKER,
    period: str = "2y",
) -> pd.DataFrame:
    """Fetch OHLCV history and cache to parquet."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise RuntimeError(f"No history returned for {ticker}")

    path = RAW_DATA_DIR / f"{ticker}_history.parquet"
    df.to_parquet(path)
    logger.info(
        "Saved %d rows of %s history to %s", len(df), ticker, path
    )
    return df


def generate_synthetic_options(
    spot: float = 585.0,
    n_expirations: int = 6,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Generate synthetic SPY-like options for offline testing."""
    from src.pricing import BlackScholes

    np.random.seed(config.RANDOM_SEED)

    r = config.FALLBACK_RISK_FREE_RATE
    q = config.FALLBACK_DIVIDEND_YIELD

    base_vol = 0.18
    skew = -0.12
    curvature = 0.30

    expiry_days = np.linspace(14, 365, n_expirations).astype(int)
    moneyness_range = np.arange(
        config.MONEYNESS_LOWER, config.MONEYNESS_UPPER + 0.01, 0.01
    )

    now = datetime.now()
    records = []

    for days in expiry_days:
        T = days / 365.0
        exp_date = now + timedelta(days=int(days))

        for m in moneyness_range:
            K = round(spot * m, 2)
            deviation = m - 1.0
            iv = base_vol + skew * deviation + curvature * deviation ** 2
            iv = np.clip(iv, config.IV_LOWER_BOUND, config.IV_UPPER_BOUND)

            for opt_type in ("call", "put"):
                bs = BlackScholes(S=spot, K=K, T=T, r=r, q=q, sigma=iv)
                price = bs.call_price() if opt_type == "call" else bs.put_price()

                spread_frac = 0.02 + 0.01 * abs(deviation)
                half_spread = price * spread_frac / 2.0
                bid = max(round(price - half_spread, 2), 0.01)
                ask = round(price + half_spread, 2)
                mid = (bid + ask) / 2.0

                intrinsic = max(
                    (spot - K) if opt_type == "call" else (K - spot), 0
                )

                records.append(
                    {
                        "contractSymbol": f"SYN{opt_type[0].upper()}{K:.0f}_{days}D",
                        "strike": K,
                        "bid": bid,
                        "ask": ask,
                        "mid_price": mid,
                        "lastPrice": mid,
                        "volume": int(np.random.exponential(500) + 10),
                        "openInterest": int(np.random.exponential(2000) + 50),
                        "impliedVolatility": iv,
                        "expiration": pd.Timestamp(exp_date.date()),
                        "option_type": opt_type,
                        "spot": spot,
                        "bid_ask_spread_pct": (ask - bid) / mid if mid > 0 else 0,
                        "moneyness": m,
                        "T": T,
                        "intrinsic_value": intrinsic,
                        "time_value": mid - intrinsic,
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d synthetic option rows (%d expirations)", len(df), n_expirations
    )
    return df, spot, r, q

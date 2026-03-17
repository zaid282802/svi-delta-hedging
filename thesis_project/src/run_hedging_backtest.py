"""Delta-hedging backtest on SPX options using 5 vol inputs.
Outputs hedging_backtest_results.parquet and hedging_summary_stats.parquet.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
Q_DIV = 0.013                    # approximate SPX dividend yield
MAX_OPTIONS_PER_REGIME = 500     # stratified sample cap per VIX regime


# build realized vol from underlying OHLC

def build_realized_vol():
    """Compute 3 realized vol estimators from SPX daily data."""
    print("computing realized vol...")

    und = pd.read_csv(RAW / "spx_underlying_2019_2024.csv")
    und["date"] = pd.to_datetime(und["date"])
    und = und.sort_values("date").reset_index(drop=True)
    print(f"  underlying data: {len(und)} trading days")
    print(f"  range: {und['date'].min().date()} to {und['date'].max().date()}")

    # close-to-close rv
    log_ret = np.log(und["close"] / und["close"].shift(1))
    und["rv_cc"] = log_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)

    # parkinson rv (high-low range)
    log_hl = np.log(und["high"] / und["low"])
    und["rv_parkinson"] = np.sqrt(
        (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2).rolling(21).mean() * TRADING_DAYS
    )

    # yang-zhang rv (OHLC, drift-independent)
    log_co = np.log(und["open"] / und["close"].shift(1))
    log_oc = np.log(und["close"] / und["open"])
    log_ho = np.log(und["high"] / und["open"])
    log_lo = np.log(und["low"] / und["open"])
    log_hc = np.log(und["high"] / und["close"])
    log_lc = np.log(und["low"] / und["close"])

    overnight_var = log_co.rolling(21).var()
    oc_var = log_oc.rolling(21).var()
    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(21).mean()
    k = 0.34 / (1.34 + 22.0 / 20.0)
    yz_var = overnight_var + k * oc_var + (1.0 - k) * rs_var
    und["rv_yangzhang"] = np.sqrt(np.maximum(yz_var, 0.0) * TRADING_DAYS)

    rv = und[["date", "close", "rv_cc", "rv_parkinson", "rv_yangzhang"]].copy()
    rv = rv.rename(columns={"close": "spot"})
    rv.to_parquet(PROCESSED / "realized_vol_daily.parquet", index=False)
    print(f"  saved realized_vol_daily.parquet")
    return rv


# load options, filter early, merge with rv and vix

def load_and_filter_options(rv_df):
    """Load raw options CSV with aggressive early filtering."""
    print("loading and filtering options...")

    usecols = [
        "date", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "impl_volatility", "delta",
        "forward_price", "optionid",
    ]

    chunks = []
    n_read = 0

    for chunk in pd.read_csv(RAW / "spx_options_2019_2024.csv",
                              usecols=usecols, chunksize=1_000_000):
        n_read += len(chunk)
        sys.stdout.write(f"\r  loading: {n_read:,} rows...")
        sys.stdout.flush()

        # heavy filtering during load to keep memory manageable
        chunk = chunk.dropna(subset=["impl_volatility", "delta"])
        chunk = chunk[chunk["best_bid"] > 0]

        # convert strike
        chunk["strike"] = chunk["strike_price"] / 1000.0

        # parse dates and compute dte
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["exdate"] = pd.to_datetime(chunk["exdate"])
        chunk["dte"] = (chunk["exdate"] - chunk["date"]).dt.days

        # keep only dte 0-95
        chunk = chunk[(chunk["dte"] >= 0) & (chunk["dte"] <= 95)]

        # filter by delta: |delta| between 0.15 and 0.85
        chunk = chunk[chunk["delta"].abs().between(0.15, 0.85)]

        # mid price floor
        chunk["mid_price"] = (chunk["best_bid"] + chunk["best_offer"]) / 2.0
        chunk = chunk[chunk["mid_price"] >= 0.10]

        # iv bounds
        chunk = chunk[(chunk["impl_volatility"] > 0.01) & (chunk["impl_volatility"] < 3.0)]

        chunks.append(chunk)

    print(f"\r  loaded {n_read:,} raw rows")

    opts = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  after early filters: {len(opts):,} rows")

    # time to expiry in years
    opts["T"] = opts["dte"] / 365.0

    # merge with spot and rv
    rv_spot = rv_df[["date", "spot", "rv_cc", "rv_parkinson", "rv_yangzhang"]].copy()
    opts = opts.merge(rv_spot, on="date", how="left")
    opts = opts.dropna(subset=["spot"])

    # moneyness
    opts["moneyness"] = opts["strike"] / opts["spot"]

    # drop strike_price (we have strike now)
    opts = opts.drop(columns=["strike_price"], errors="ignore")

    print(f"  after merge with underlying: {len(opts):,} rows")
    return opts


def add_vix_and_rates(opts):
    """Add VIX close, VIX regime, and risk-free rate."""
    print("  adding vix regime...")

    # vix
    vix = pd.read_csv(RAW / "VIX_History.csv")
    vix["date"] = pd.to_datetime(vix["DATE"], format="mixed")
    vix = vix[["date", "CLOSE"]].rename(columns={"CLOSE": "vix_close"})
    vix = vix.sort_values("date").drop_duplicates("date")
    opts = opts.merge(vix, on="date", how="left")
    opts["vix_close"] = opts["vix_close"].ffill()

    opts["vix_regime"] = pd.cut(
        opts["vix_close"],
        bins=[0, 15, 25, 35, 200],
        labels=["Low", "Normal", "High", "Crisis"],
    )

    # risk-free rate
    print("  adding risk-free rates...")
    rf = pd.read_csv(RAW / "risk_free_rates_2019_2024.csv")
    rf["date"] = pd.to_datetime(rf["date"])
    rf["rate"] = rf["rate"] / 100.0

    # for each date, get the 30-day rate as a reasonable approximation
    rf_30 = rf[rf["days"].between(25, 35)].groupby("date")["rate"].mean().reset_index()
    rf_30 = rf_30.rename(columns={"rate": "rf_rate"})
    opts = opts.merge(rf_30, on="date", how="left")
    opts["rf_rate"] = opts["rf_rate"].ffill().bfill().fillna(0.043)

    return opts


def compute_atm_iv(opts):
    """Compute ATM IV per (date, exdate) and assign svi_implied_vol."""
    print("  computing atm (flat bsm) iv per date x expiry...")

    # atm iv: for each (date, exdate), find the iv closest to moneyness=1.0
    opts["abs_m_diff"] = (opts["moneyness"] - 1.0).abs()
    atm = (
        opts.sort_values("abs_m_diff")
        .groupby(["date", "exdate"])
        .first()["impl_volatility"]
        .reset_index()
        .rename(columns={"impl_volatility": "flat_bsm_iv"})
    )
    opts = opts.merge(atm, on=["date", "exdate"], how="left")
    opts = opts.drop(columns=["abs_m_diff"])

    # svi surface iv = the option's own market iv at its specific strike
    opts["svi_implied_vol"] = opts["impl_volatility"]

    return opts


# select and sample options for hedging

def select_hedging_sample(opts):
    """Select options suitable for delta-hedging backtest."""
    print("selecting hedging sample...")

    opts = opts.sort_values(["optionid", "date"]).reset_index(drop=True)

    # for each option, compute summary stats
    first_obs = opts.groupby("optionid").agg(
        first_date=("date", "first"),
        last_date=("date", "last"),
        exdate=("exdate", "first"),
        first_dte=("dte", "first"),
        last_dte=("dte", "last"),
        n_obs=("date", "count"),
        cp_flag=("cp_flag", "first"),
        strike=("strike", "first"),
        first_delta=("delta", "first"),
        first_moneyness=("moneyness", "first"),
        first_vix=("vix_close", "first"),
        first_vix_regime=("vix_regime", "first"),
    ).reset_index()

    n0 = len(first_obs)
    print(f"  unique options: {n0:,}")

    # entry dte between 14 and 90
    mask = (first_obs["first_dte"] >= 14) & (first_obs["first_dte"] <= 90)
    first_obs = first_obs[mask]
    print(f"  after dte [14, 90]: {len(first_obs):,}")

    # near-atm: |delta| between 0.25 and 0.75
    mask = first_obs["first_delta"].abs().between(0.25, 0.75)
    first_obs = first_obs[mask]
    print(f"  after |delta| [0.25, 0.75]: {len(first_obs):,}")

    # must observe to near-expiry (last_dte <= 2)
    mask = first_obs["last_dte"] <= 2
    first_obs = first_obs[mask]
    print(f"  after expiry coverage (last_dte <= 2): {len(first_obs):,}")

    # at least 10 daily observations
    mask = first_obs["n_obs"] >= 10
    first_obs = first_obs[mask]
    print(f"  after min obs >= 10: {len(first_obs):,}")

    # continuity check: n_obs should be at least 60% of dte
    first_obs["obs_ratio"] = first_obs["n_obs"] / (first_obs["first_dte"] / 1.4)
    mask = first_obs["obs_ratio"] >= 0.6
    first_obs = first_obs[mask]
    print(f"  after continuity filter: {len(first_obs):,}")

    # moneyness bucket
    first_obs["moneyness_bucket"] = pd.cut(
        first_obs["first_moneyness"],
        bins=[0.0, 0.95, 0.99, 1.01, 1.05, 2.0],
        labels=["deep_otm_put", "otm_put", "atm", "otm_call", "deep_otm_call"],
    )

    # maturity bucket
    first_obs["maturity_bucket"] = pd.cut(
        first_obs["first_dte"],
        bins=[0, 30, 60, 91],
        labels=["short", "medium", "long"],
    )

    return first_obs


def stratified_sample(first_obs, max_per_regime=MAX_OPTIONS_PER_REGIME):
    """Stratified random sample across VIX regimes."""
    print(f"  stratified sampling (max {max_per_regime} per vix regime):")
    print(f"  before: {len(first_obs):,}")
    print(first_obs["first_vix_regime"].value_counts().to_string())

    frames = []
    for regime, grp in first_obs.groupby("first_vix_regime", observed=True):
        n_take = min(len(grp), max_per_regime)
        frames.append(grp.sample(n=n_take, random_state=42))
    sampled = pd.concat(frames, ignore_index=True)
    print(f"  after: {len(sampled):,}")
    print(sampled["first_vix_regime"].value_counts().to_string())
    return sampled


# delta-hedging simulation

def bsm_delta_vec(S, K, T, r, sigma, cp_flag, q=Q_DIV):
    """Vectorized BSM delta."""
    T = np.maximum(np.asarray(T, dtype=np.float64), 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    S = np.asarray(S, dtype=np.float64)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if cp_flag == "C":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def hedge_one_option(odata, cp_flag):
    """Delta-hedge one option using 5 vol inputs, returns pnl dict."""
    n = len(odata)
    if n < 2:
        return None

    S = odata["spot"].values.astype(np.float64)
    K = float(odata["strike"].iloc[0])
    V = odata["mid_price"].values.astype(np.float64)
    T = odata["T"].values.astype(np.float64)
    r = odata["rf_rate"].values.astype(np.float64)

    # dS and dV
    dS = np.diff(S)      # S[t+1] - S[t], length n-1
    dV = np.diff(V)      # V[t+1] - V[t], length n-1

    vol_map = {
        "flat_bsm":    odata["flat_bsm_iv"].values,
        "svi_surface": odata["svi_implied_vol"].values,
        "rv_cc":       odata["rv_cc"].values,
        "rv_park":     odata["rv_parkinson"].values,
        "rv_yz":       odata["rv_yangzhang"].values,
    }

    results = {}
    for vol_name, sigma_arr in vol_map.items():
        sigma = pd.Series(sigma_arr, dtype=np.float64).ffill().bfill().values
        if np.all(np.isnan(sigma)):
            results[f"pnl_{vol_name}"] = np.nan
            continue
        sigma = np.maximum(sigma, 0.01)

        # compute delta at each observation date
        deltas = bsm_delta_vec(S, K, T, r, sigma, cp_flag)

        # hedging error = sum(delta_t * dS_t - dV_t)
        hedge_pnl = np.sum(deltas[:-1] * dS - dV)
        results[f"pnl_{vol_name}"] = float(hedge_pnl)

    return results


def run_backtest(opts, sample_df):
    """Run delta-hedging backtest for all sampled options."""
    print(f"running hedging backtest ({len(sample_df):,} options)...")

    option_ids = sample_df["optionid"].values

    # pre-filter opts to only include sampled options
    opts_subset = opts[opts["optionid"].isin(option_ids)].copy()
    opts_subset = opts_subset.sort_values(["optionid", "date"])
    print(f"  working with {len(opts_subset):,} daily observations")

    # group once
    grouped = dict(list(opts_subset.groupby("optionid")))

    results_list = []
    t_start = time.time()

    for i, oid in enumerate(option_ids):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            meta = sample_df[sample_df["optionid"] == oid].iloc[0]
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(option_ids) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(option_ids)} "
                  f"(date: {meta['first_date'].strftime('%Y-%m-%d')}) "
                  f"[{rate:.0f} opt/s, ETA {eta:.0f}s]")

        odata = grouped.get(oid)
        if odata is None or len(odata) < 2:
            continue

        cp = odata["cp_flag"].iloc[0]
        pnl = hedge_one_option(odata, cp)
        if pnl is None:
            continue

        meta = sample_df[sample_df["optionid"] == oid].iloc[0]

        row = {
            "optionid": oid,
            "entry_date": meta["first_date"],
            "expiry_date": meta["exdate"],
            "strike": meta["strike"],
            "cp_flag": cp,
            "entry_dte": meta["first_dte"],
            "entry_spot": odata["spot"].iloc[0],
            "moneyness_bucket": str(meta["moneyness_bucket"]),
            "maturity_bucket": str(meta["maturity_bucket"]),
            "vix_regime_at_entry": str(meta["first_vix_regime"]),
            "entry_vix": meta["first_vix"],
            "n_hedge_days": len(odata),
        }
        row.update(pnl)
        results_list.append(row)

    results = pd.DataFrame(results_list)
    elapsed = time.time() - t_start
    print(f"  backtest done: {len(results):,} options in {elapsed:.1f}s")
    return results


# summary statistics

def compute_summary(results):
    """Compute summary stats by vol_input x moneyness x maturity x vix regime."""
    print("computing summary stats...")

    pnl_cols = [c for c in results.columns if c.startswith("pnl_")]
    vol_names = [c.replace("pnl_", "") for c in pnl_cols]

    benchmark_std = results["pnl_flat_bsm"].std()
    group_cols = ["moneyness_bucket", "maturity_bucket", "vix_regime_at_entry"]

    def stats_for_group(vals, bm_std):
        """Standard hedging performance metrics."""
        if len(vals) < 3:
            return None
        return {
            "n_options": len(vals),
            "mean_pnl": vals.mean(),
            "std_pnl": vals.std(),
            "median_pnl": vals.median(),
            "mae": vals.abs().mean(),
            "rmse": np.sqrt((vals ** 2).mean()),
            "gain_vs_flat_bsm": (1 - vals.std() / bm_std) * 100 if bm_std > 0 else np.nan,
            "pct_positive": (vals > 0).mean() * 100,
            "p5": vals.quantile(0.05),
            "p95": vals.quantile(0.95),
        }

    rows = []

    # overall stats
    for vn in vol_names:
        vals = results[f"pnl_{vn}"].dropna()
        s = stats_for_group(vals, benchmark_std)
        if s:
            s.update({"vol_input": vn, "moneyness_bucket": "ALL",
                       "maturity_bucket": "ALL", "vix_regime": "ALL"})
            rows.append(s)

    # grouped stats
    for keys, grp in results.groupby(group_cols, observed=False):
        mb, mtb, vr = keys
        bm_std_grp = grp["pnl_flat_bsm"].dropna().std()
        for vn in vol_names:
            vals = grp[f"pnl_{vn}"].dropna()
            s = stats_for_group(vals, bm_std_grp)
            if s:
                s.update({"vol_input": vn, "moneyness_bucket": mb,
                           "maturity_bucket": mtb, "vix_regime": vr})
                rows.append(s)

    summary = pd.DataFrame(rows)

    # reorder columns
    first_cols = ["vol_input", "moneyness_bucket", "maturity_bucket", "vix_regime"]
    other_cols = [c for c in summary.columns if c not in first_cols]
    summary = summary[first_cols + other_cols]

    return summary


def print_results(results, summary):
    """Print key results to console."""
    pnl_cols = [c for c in results.columns if c.startswith("pnl_")]

    print(f"\ntotal options hedged: {len(results):,}")
    print(f"date range: {results['entry_date'].min().date()} to "
          f"{results['entry_date'].max().date()}")

    print(f"\nsample composition:")
    print(f"  by vix regime:  {results['vix_regime_at_entry'].value_counts().to_dict()}")
    print(f"  by type:        {results['cp_flag'].value_counts().to_dict()}")
    print(f"  by moneyness:   {results['moneyness_bucket'].value_counts().to_dict()}")
    print(f"  by maturity:    {results['maturity_bucket'].value_counts().to_dict()}")

    print("\noverall hedging error ($ per option):")
    print(f"{'vol_input':>15s}  {'mean':>8s}  {'std':>8s}  {'median':>8s}  "
          f"{'mae':>8s}  {'rmse':>8s}  {'gain%':>7s}")

    benchmark_std = results["pnl_flat_bsm"].std()
    for col in pnl_cols:
        vn = col.replace("pnl_", "")
        vals = results[col].dropna()
        gain = (1 - vals.std() / benchmark_std) * 100
        print(f"{vn:>15s}  {vals.mean():8.2f}  {vals.std():8.2f}  "
              f"{vals.median():8.2f}  {vals.abs().mean():8.2f}  "
              f"{np.sqrt((vals**2).mean()):8.2f}  {gain:+6.1f}%")

    print("\nmean hedging error by vix regime:")
    regime_means = results.groupby("vix_regime_at_entry", observed=True)[pnl_cols].mean()
    regime_stds = results.groupby("vix_regime_at_entry", observed=True)[pnl_cols].std()
    print("mean pnl:")
    print(regime_means.round(2).to_string())
    print("\nstd pnl:")
    print(regime_stds.round(2).to_string())

    print("\nmean hedging error by moneyness:")
    mono_stats = results.groupby("moneyness_bucket", observed=True)[pnl_cols].agg(["mean", "std"])
    print(mono_stats.round(2).to_string())

    print("\ngain vs flat bsm by vix regime:")
    for regime in sorted(results["vix_regime_at_entry"].unique()):
        grp = results[results["vix_regime_at_entry"] == regime]
        bm = grp["pnl_flat_bsm"].std()
        gains = {col.replace("pnl_", ""): (1 - grp[col].dropna().std() / bm) * 100
                 for col in pnl_cols}
        print(f"  {regime}: {gains}")


if __name__ == "__main__":
    t0 = time.time()

    # check if processed data already exists (skip expensive reload)
    spx_parquet = PROCESSED / "spx_with_svi.parquet"
    if spx_parquet.exists():
        print("loading cached spx_with_svi.parquet...")
        opts = pd.read_parquet(spx_parquet)
        print(f"  loaded {len(opts):,} rows")

        rv_parquet = PROCESSED / "realized_vol_daily.parquet"
        if not rv_parquet.exists():
            build_realized_vol()
    else:
        rv_df = build_realized_vol()

        opts = load_and_filter_options(rv_df)
        opts = add_vix_and_rates(opts)
        opts = compute_atm_iv(opts)

        opts.to_parquet(spx_parquet, index=False)
        print(f"  saved spx_with_svi.parquet ({len(opts):,} rows)")

    # select sample
    sample = select_hedging_sample(opts)

    if len(sample) > MAX_OPTIONS_PER_REGIME * 4:
        sample = stratified_sample(sample)
    else:
        print(f"  using all {len(sample):,} qualifying options (no sampling needed)")

    # run backtest
    results = run_backtest(opts, sample)

    results.to_parquet(PROCESSED / "hedging_backtest_results.parquet", index=False)
    print(f"  saved hedging_backtest_results.parquet ({len(results):,} rows)")

    # summary
    summary = compute_summary(results)
    summary.to_parquet(PROCESSED / "hedging_summary_stats.parquet", index=False)
    print(f"  saved hedging_summary_stats.parquet ({len(summary):,} rows)")

    print_results(results, summary)

    elapsed = time.time() - t0
    print(f"\ntotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

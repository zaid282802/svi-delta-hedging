"""Durrleman butterfly arbitrage check on SVI calibration results."""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, minimize

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
Q_DIV = 0.013


# svi functions (same as run_svi_calibration.py)

def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


BOUNDS_LO = np.array([-0.5, 0.001, -0.99, -0.5, 0.001])
BOUNDS_HI = np.array([0.5, 2.0, 0.99, 0.5, 2.0])
OPT_BOUNDS = list(zip(BOUNDS_LO, BOUNDS_HI))


def calibrate_one_slice(strikes, market_ivs, forward, T):
    """LS warm-start -> DE -> L-BFGS-B polish."""
    if len(strikes) < 5 or T <= 0 or forward <= 0:
        return None

    strikes = np.asarray(strikes, dtype=np.float64)
    market_ivs = np.asarray(market_ivs, dtype=np.float64)
    k = np.log(strikes / forward)
    market_w = market_ivs ** 2 * T

    def residuals(params):
        return svi_total_variance(k, *params) - market_w

    def objective(params):
        return np.sum(residuals(params) ** 2)

    try:
        atm_w = float(np.interp(0, k, market_w))
        x0 = [atm_w, 0.1, -0.3, 0.0, 0.1]

        r_ls = least_squares(residuals, x0, bounds=(BOUNDS_LO, BOUNDS_HI),
                              method="trf", max_nfev=200)
        r_de = differential_evolution(objective, OPT_BOUNDS, x0=r_ls.x,
                                       maxiter=15, seed=42, tol=1e-10,
                                       polish=False, popsize=5)
        r_local = minimize(objective, x0=r_de.x, method="L-BFGS-B", bounds=OPT_BOUNDS)

        best_params = r_local.x
        best_fun = r_local.fun

        w_fit = svi_total_variance(k, *best_params)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
        rmse_bps = float(np.sqrt(np.mean((iv_fit - market_ivs) ** 2)) * 10_000)

        if rmse_bps > 200.0:
            for rho0 in [-0.5, 0.0, -0.7]:
                x0_alt = [atm_w, 0.05, rho0, 0.0, 0.2]
                try:
                    r2 = least_squares(residuals, x0_alt, bounds=(BOUNDS_LO, BOUNDS_HI),
                                        method="trf", max_nfev=200)
                    r3 = minimize(objective, x0=r2.x, method="L-BFGS-B", bounds=OPT_BOUNDS)
                    if r3.fun < best_fun:
                        best_params = r3.x
                        best_fun = r3.fun
                except Exception:
                    continue
            w_fit = svi_total_variance(k, *best_params)
            iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
            rmse_bps = float(np.sqrt(np.mean((iv_fit - market_ivs) ** 2)) * 10_000)

        a, b, rho, m, sigma = best_params
        return (float(a), float(b), float(rho), float(m), float(sigma), rmse_bps)
    except Exception:
        return None


# durrleman butterfly arbitrage condition

def durrleman_condition(a, b, rho, m, sigma, k_grid):
    """Evaluate g(k) >= 0 condition from Durrleman (2005)."""
    k_grid = np.asarray(k_grid, dtype=np.float64)

    # w(k) and derivatives
    diff = k_grid - m
    root = np.sqrt(diff ** 2 + sigma ** 2)

    w = a + b * (rho * diff + root)

    # w'(k) = b * (rho + diff / root)
    w_prime = b * (rho + diff / root)

    # w''(k) = b * sigma^2 / root^3
    w_double_prime = b * sigma ** 2 / root ** 3

    # guard against w <= 0
    w_safe = np.maximum(w, 1e-12)

    # g(k)
    term1 = (1.0 - k_grid * w_prime / (2.0 * w_safe)) ** 2
    term2 = (w_prime ** 2) / 4.0 * (1.0 / w_safe + 0.25)
    term3 = w_double_prime / 2.0

    g = term1 - term2 + term3

    min_g = float(np.min(g))
    is_arb_free = bool(min_g >= -1e-10)

    return g, min_g, is_arb_free


def main():
    t0 = time.time()

    # load data and identify slices
    print("loading data and identifying calibration slices...")

    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet")
    results = pd.read_parquet(PROCESSED / "hedging_backtest_results.parquet")
    sampled_ids = set(results["optionid"].values)

    sampled_mask = opts["optionid"].isin(sampled_ids)
    sampled_rows = opts[sampled_mask]
    all_slices = sampled_rows[["date", "exdate"]].drop_duplicates()

    # thin to every 5th trading day per exdate (same grid as run_svi_calibration.py)
    cal_points = set()
    for ex, grp in all_slices.groupby("exdate"):
        dates = sorted(grp["date"].unique())
        for i in range(0, len(dates), 5):
            cal_points.add((dates[i], ex))
        cal_points.add((dates[0], ex))
        cal_points.add((dates[-1], ex))

    cal_points = sorted(cal_points)
    print(f"  calibration points: {len(cal_points)}")

    # prepare calibration data
    opts["_key"] = list(zip(opts["date"], opts["exdate"]))
    cal_data = opts[opts["_key"].isin(set(cal_points))].copy()
    opts.drop(columns=["_key"], inplace=True)
    cal_data.drop(columns=["_key"], inplace=True)

    tasks = []
    for (dt, ex), grp in cal_data.groupby(["date", "exdate"]):
        strikes = grp["strike"].values
        ivs = grp["impl_volatility"].values
        T = float(grp["T"].iloc[0])
        fwd_vals = grp["forward_price"].dropna()
        if len(fwd_vals) > 0:
            forward = float(fwd_vals.iloc[0])
        else:
            spot = float(grp["spot"].iloc[0])
            r = float(grp["rf_rate"].iloc[0])
            forward = spot * np.exp((r - Q_DIV) * T)
        tasks.append(((dt, ex), strikes, ivs, forward, T))

    print(f"  calibration tasks: {len(tasks)}")
    del cal_data

    # run svi calibration
    print(f"running SVI calibration ({len(tasks)} slices)...")

    MAX_RMSE_BPS = 300.0
    cal_rows = []
    n_ok = 0
    n_fail = 0

    t_cal = time.time()
    k_grid = np.linspace(-0.5, 0.5, 201)

    for i, (slice_key, strikes, ivs, forward, T) in enumerate(tasks):
        dt, ex = slice_key
        result = calibrate_one_slice(strikes, ivs, forward, T)

        if result is not None and result[5] <= MAX_RMSE_BPS:
            a, b, rho, m, sigma, rmse_bps = result

            # durrleman butterfly check
            g_vals, min_g, arb_free = durrleman_condition(a, b, rho, m, sigma, k_grid)

            cal_rows.append({
                "date": dt,
                "exdate": ex,
                "T": T,
                "forward": forward,
                "n_strikes": len(strikes),
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "rmse_bps": rmse_bps,
                "arb_free": arb_free,
                "min_g": min_g,
            })
            n_ok += 1
        else:
            n_fail += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - t_cal
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            n_arb_free = sum(1 for r in cal_rows if r["arb_free"])
            print(f"  {i+1}/{len(tasks)} slices "
                  f"({n_ok} ok, {n_fail} fail) "
                  f"[{rate:.1f}/s, ETA {eta:.0f}s] "
                  f"arb-free: {n_arb_free}/{n_ok}")

    cal_time = time.time() - t_cal

    # save results
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_parquet(PROCESSED / "svi_calibration_results.parquet", index=False)

    print(f"  calibration done in {cal_time:.1f}s ({cal_time/60:.1f} min)")
    print(f"  saved svi_calibration_results.parquet ({len(cal_df)} rows)")

    # summary
    print(f"\n  total slices calibrated:  {len(cal_df)}")
    print(f"  failed / rejected:        {n_fail}")

    n_total = len(cal_df)
    n_arb_free = cal_df["arb_free"].sum()
    n_arb_viol = n_total - n_arb_free

    print(f"  rmse (bps):")
    print(f"    median:  {cal_df['rmse_bps'].median():.1f}")
    print(f"    mean:    {cal_df['rmse_bps'].mean():.1f}")
    print(f"    p95:     {cal_df['rmse_bps'].quantile(0.95):.1f}")
    print(f"    max:     {cal_df['rmse_bps'].max():.1f}")

    print(f"  durrleman butterfly arbitrage check:")
    print(f"  grid: 201 points in [-0.5, 0.5], tol: g(k) >= -1e-10")
    print(f"  arbitrage-free:   {n_arb_free:,} / {n_total:,}  "
          f"({n_arb_free/n_total*100:.1f}%)")
    print(f"  violations:       {n_arb_viol:,} / {n_total:,}  "
          f"({n_arb_viol/n_total*100:.1f}%)")

    if n_arb_viol > 0:
        viol = cal_df[~cal_df["arb_free"]]
        print(f"  violation severity (min g across violating slices):")
        print(f"    min:     {viol['min_g'].min():.6f}")
        print(f"    median:  {viol['min_g'].median():.6f}")
        print(f"    p5:      {viol['min_g'].quantile(0.05):.6f}")

        # breakdown by dte bucket
        cal_df["dte"] = (cal_df["T"] * 365).round().astype(int)
        cal_df["dte_bucket"] = pd.cut(cal_df["dte"], bins=[0, 30, 60, 90, 365],
                                       labels=["0-30", "31-60", "61-90", "91+"])
        print(f"  arb-free rate by DTE bucket:")
        for bucket, grp in cal_df.groupby("dte_bucket", observed=True):
            pct = grp["arb_free"].mean() * 100
            print(f"    {bucket}: {grp['arb_free'].sum()}/{len(grp)} ({pct:.1f}%)")

        # breakdown by rmse bucket
        cal_df["rmse_bucket"] = pd.cut(cal_df["rmse_bps"],
                                        bins=[0, 25, 50, 100, 200, 300],
                                        labels=["0-25", "25-50", "50-100", "100-200", "200-300"])
        print(f"  arb-free rate by RMSE bucket:")
        for bucket, grp in cal_df.groupby("rmse_bucket", observed=True):
            pct = grp["arb_free"].mean() * 100
            print(f"    {bucket} bps: {grp['arb_free'].sum()}/{len(grp)} ({pct:.1f}%)")

    # svi parameter stats
    print(f"  svi parameter statistics:")
    for param in ["a", "b", "rho", "m", "sigma"]:
        vals = cal_df[param]
        print(f"  {param:>5s}:  mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"[{vals.quantile(0.05):+.4f}, {vals.quantile(0.95):+.4f}]")

    total_time = time.time() - t0
    print(f"  total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()

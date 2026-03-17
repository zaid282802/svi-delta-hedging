"""SVI calibration and hedging backtest recomputation."""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, least_squares, minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
Q_DIV = 0.013


# svi calibration

def svi_total_variance(k, a, b, rho, m, sigma):
    """Gatheral raw SVI total variance."""
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


def svi_iv_from_params(k, T, a, b, rho, m, sigma):
    """Convert SVI total variance to implied vol."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0.0) / max(T, 1e-10))


BOUNDS_LO = np.array([-0.5, 0.001, -0.99, -0.5, 0.001])
BOUNDS_HI = np.array([0.5, 2.0, 0.99, 0.5, 2.0])
OPT_BOUNDS = list(zip(BOUNDS_LO, BOUNDS_HI))


def calibrate_one_slice(strikes, market_ivs, forward, T):
    """Calibrate raw SVI to one expiry slice, returns params + rmse or None."""
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
        # least_squares warm-start
        atm_w = float(np.interp(0, k, market_w))
        x0 = [atm_w, 0.1, -0.3, 0.0, 0.1]

        r_ls = least_squares(
            residuals, x0,
            bounds=(BOUNDS_LO, BOUNDS_HI),
            method="trf",
            max_nfev=200,
        )

        # differential evolution seeded with LS solution
        r_de = differential_evolution(
            objective,
            OPT_BOUNDS,
            x0=r_ls.x,
            maxiter=15,
            seed=42,
            tol=1e-10,
            polish=False,
            popsize=5,
        )

        # L-BFGS-B polish
        r_local = minimize(
            objective,
            x0=r_de.x,
            method="L-BFGS-B",
            bounds=OPT_BOUNDS,
        )

        best_params = r_local.x
        best_fun = r_local.fun

        # check fit quality, retry with different starts if poor
        w_fit = svi_total_variance(k, *best_params)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
        rmse_bps = float(np.sqrt(np.mean((iv_fit - market_ivs) ** 2)) * 10_000)

        if rmse_bps > 200.0:
            for rho0 in [-0.5, 0.0, -0.7]:
                x0_alt = [atm_w, 0.05, rho0, 0.0, 0.2]
                try:
                    r2 = least_squares(residuals, x0_alt,
                                       bounds=(BOUNDS_LO, BOUNDS_HI),
                                       method="trf", max_nfev=200)
                    r3 = minimize(objective, x0=r2.x, method="L-BFGS-B",
                                  bounds=OPT_BOUNDS)
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


# main pipeline

def main():
    t0 = time.time()

    # load data
    print("loading data...")

    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet")
    results_old = pd.read_parquet(PROCESSED / "hedging_backtest_results.parquet")
    sampled_ids = set(results_old["optionid"].values)

    print(f"  full options data: {len(opts):,} rows")
    print(f"  sampled options: {len(sampled_ids)}")

    # build calibration grid
    print("building calibration grid...")

    sampled_mask = opts["optionid"].isin(sampled_ids)
    sampled_rows = opts[sampled_mask]
    all_slices = sampled_rows[["date", "exdate"]].drop_duplicates()
    print(f"  total (date, exdate) slices needed: {len(all_slices)}")
    print(f"  unique exdates: {all_slices['exdate'].nunique()}")

    # thin: calibrate every 5th trading day per exdate
    cal_points = set()
    exdate_dates = {}  # exdate -> sorted list of all dates
    for ex, grp in all_slices.groupby("exdate"):
        dates = sorted(grp["date"].unique())
        exdate_dates[ex] = dates
        for i in range(0, len(dates), 5):
            cal_points.add((dates[i], ex))
        # always include first and last
        cal_points.add((dates[0], ex))
        cal_points.add((dates[-1], ex))

    cal_points = sorted(cal_points)
    print(f"  thinned calibration points (every 5 days): {len(cal_points)}")

    # prepare calibration data
    print("preparing calibration data...")

    # fast lookup for (date, exdate) -> slice data
    cal_set = set(cal_points)
    opts["_key"] = list(zip(opts["date"], opts["exdate"]))
    cal_data = opts[opts["_key"].isin(cal_set)].copy()
    opts.drop(columns=["_key"], inplace=True)
    cal_data.drop(columns=["_key"], inplace=True)
    print(f"  option rows for calibration: {len(cal_data):,}")

    # group and build task list
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
    print(f"calibrating svi ({len(tasks)} slices)...")

    # (date, exdate) -> (a, b, rho, m, sigma, rmse_bps)
    svi_params = {}
    n_ok = 0
    n_fail = 0
    rmse_list = []

    t_cal = time.time()

    MAX_RMSE_BPS = 300.0  # reject fits worse than this

    for i, (slice_key, strikes, ivs, forward, T) in enumerate(tasks):
        result = calibrate_one_slice(strikes, ivs, forward, T)

        if result is not None and result[5] <= MAX_RMSE_BPS:
            svi_params[slice_key] = result
            n_ok += 1
            rmse_list.append(result[5])
        else:
            n_fail += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - t_cal
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            med_rmse = np.median(rmse_list) if rmse_list else 0
            print(f"  {i+1}/{len(tasks)} slices "
                  f"({n_ok} ok, {n_fail} fail) "
                  f"[{rate:.1f} slices/s, ETA {eta:.0f}s] "
                  f"median RMSE: {med_rmse:.1f} bps")

    cal_time = time.time() - t_cal
    print(f"  calibration done in {cal_time:.1f}s ({cal_time/60:.1f} min)")
    print(f"  success: {n_ok}/{len(tasks)} ({n_ok/len(tasks)*100:.1f}%)")
    if rmse_list:
        print(f"  RMSE (bps): median={np.median(rmse_list):.1f}, "
              f"mean={np.mean(rmse_list):.1f}, "
              f"p95={np.percentile(rmse_list, 95):.1f}, "
              f"max={np.max(rmse_list):.1f}")

    # interpolate svi params for non-calibrated dates
    print("interpolating svi params for all dates...")

    # for each exdate, linearly interpolate each SVI parameter
    full_svi = {}

    for ex, dates in exdate_dates.items():
        # collect calibrated params for this exdate
        cal_dates = []
        cal_params_list = []
        for dt in dates:
            key = (dt, ex)
            if key in svi_params:
                cal_dates.append(dt)
                cal_params_list.append(svi_params[key][:5])  # (a,b,rho,m,sigma)

        if len(cal_dates) == 0:
            continue

        if len(cal_dates) == 1:
            # only one calibration point, use for all dates
            params = cal_params_list[0]
            for dt in dates:
                full_svi[(dt, ex)] = params
        else:
            # interpolate each parameter
            cal_ordinals = np.array([d.toordinal() for d in cal_dates])
            param_array = np.array(cal_params_list)  # shape (n_cal, 5)

            all_ordinals = np.array([d.toordinal() for d in dates])

            for p_idx in range(5):
                interp_func = interp1d(
                    cal_ordinals, param_array[:, p_idx],
                    kind="linear", fill_value="extrapolate", bounds_error=False,
                )
                interp_vals = interp_func(all_ordinals)

                for j, dt in enumerate(dates):
                    key = (dt, ex)
                    if key not in full_svi:
                        full_svi[key] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    full_svi[key][p_idx] = float(interp_vals[j])

    print(f"  full svi params for {len(full_svi)} (date, exdate) pairs")
    print(f"  coverage: {len(full_svi)/len(all_slices)*100:.1f}% of needed slices")

    # compute svi ivs for sampled options
    print("computing svi-calibrated ivs for sampled options...")

    sampled_opts = opts[sampled_mask].copy()
    new_svi_iv = np.full(len(sampled_opts), np.nan)

    n_updated = 0
    n_fallback = 0

    MAX_IV_DEVIATION = 0.05  # reject SVI IV if it deviates > 5 vol points from market

    for i, (idx, row) in enumerate(sampled_opts.iterrows()):
        sk = (row["date"], row["exdate"])
        params = full_svi.get(sk)

        if params is not None:
            a, b, rho, m, sigma = params
            T = row["T"]
            fwd = row["forward_price"]
            if pd.isna(fwd):
                fwd = row["spot"] * np.exp((row["rf_rate"] - Q_DIV) * T)
            log_k = np.log(row["strike"] / fwd)
            svi_iv = svi_iv_from_params(np.array([log_k]), T, a, b, rho, m, sigma)[0]
            market_iv = row["impl_volatility"]
            # accept svi iv if reasonable
            if (0.01 <= svi_iv <= 3.0
                    and abs(svi_iv - market_iv) <= MAX_IV_DEVIATION):
                new_svi_iv[i] = svi_iv
                n_updated += 1
            else:
                new_svi_iv[i] = market_iv
                n_fallback += 1
        else:
            new_svi_iv[i] = row["impl_volatility"]
            n_fallback += 1

    sampled_opts["svi_implied_vol"] = new_svi_iv

    print(f"  svi-calibrated: {n_updated:,} observations")
    print(f"  fallback to market iv: {n_fallback:,} observations")

    # comparison
    valid = sampled_opts.dropna(subset=["svi_implied_vol"])
    diff = valid["svi_implied_vol"] - valid["impl_volatility"]
    print(f"  svi iv vs market iv difference:")
    print(f"    mean:    {diff.mean()*10000:+.1f} bps")
    print(f"    std:      {diff.std()*10000:.1f} bps")
    print(f"    max abs:  {diff.abs().max()*10000:.1f} bps")
    print(f"    median:  {diff.median()*10000:+.1f} bps")

    # save updated parquet
    opts.loc[sampled_mask, "svi_implied_vol"] = sampled_opts["svi_implied_vol"].values
    opts.to_parquet(PROCESSED / "spx_with_svi.parquet", index=False)
    print(f"  saved updated spx_with_svi.parquet")

    # rerun hedging backtest
    print("rerunning delta-hedging backtest...")

    from run_hedging_backtest import (
        hedge_one_option,
        compute_summary,
        print_results,
    )

    sampled_opts_sorted = sampled_opts.sort_values(["optionid", "date"])
    grouped = dict(list(sampled_opts_sorted.groupby("optionid")))

    results_list = []
    t_bt = time.time()

    option_ids = results_old["optionid"].values

    for i, oid in enumerate(option_ids):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  hedging option {i+1} of {len(option_ids)}")

        odata = grouped.get(oid)
        if odata is None or len(odata) < 2:
            continue

        cp = odata["cp_flag"].iloc[0]
        pnl = hedge_one_option(odata, cp)
        if pnl is None:
            continue

        old_row = results_old[results_old["optionid"] == oid].iloc[0]

        row = {
            "optionid": oid,
            "entry_date": old_row["entry_date"],
            "expiry_date": old_row["expiry_date"],
            "strike": old_row["strike"],
            "cp_flag": cp,
            "entry_dte": old_row["entry_dte"],
            "entry_spot": old_row["entry_spot"],
            "moneyness_bucket": old_row["moneyness_bucket"],
            "maturity_bucket": old_row["maturity_bucket"],
            "vix_regime_at_entry": old_row["vix_regime_at_entry"],
            "entry_vix": old_row["entry_vix"],
            "n_hedge_days": len(odata),
        }
        row.update(pnl)
        results_list.append(row)

    results = pd.DataFrame(results_list)
    bt_time = time.time() - t_bt
    print(f"  backtest complete: {len(results):,} options in {bt_time:.1f}s")

    # save and report
    results.to_parquet(PROCESSED / "hedging_backtest_results.parquet", index=False)
    print(f"  saved hedging_backtest_results.parquet ({len(results):,} rows)")

    summary = compute_summary(results)
    summary.to_parquet(PROCESSED / "hedging_summary_stats.parquet", index=False)
    print(f"  saved hedging_summary_stats.parquet ({len(summary):,} rows)")

    print_results(results, summary)

    # compare old vs new results
    print("comparing before (market iv) vs after (svi calibrated)...")

    merged = results_old[["optionid", "pnl_svi_surface"]].merge(
        results[["optionid", "pnl_svi_surface"]],
        on="optionid", suffixes=("_before", "_after"),
    )

    diff_pnl = merged["pnl_svi_surface_after"] - merged["pnl_svi_surface_before"]
    print(f"  svi surface pnl change (after - before):")
    print(f"    mean:    ${diff_pnl.mean():.2f}")
    print(f"    std:     ${diff_pnl.std():.2f}")
    print(f"    max abs: ${diff_pnl.abs().max():.2f}")

    bm_std = results["pnl_flat_bsm"].std()
    old_svi_std = results_old["pnl_svi_surface"].std()
    new_svi_std = results["pnl_svi_surface"].std()
    print(f"  std of hedging error:")
    print(f"    flat bsm benchmark:          ${bm_std:.2f}")
    print(f"    svi (market iv shortcut):     ${old_svi_std:.2f}  "
          f"(gain: {(1-old_svi_std/bm_std)*100:+.1f}%)")
    print(f"    svi (proper calibration):     ${new_svi_std:.2f}  "
          f"(gain: {(1-new_svi_std/bm_std)*100:+.1f}%)")

    total_time = time.time() - t0
    print(f"total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()

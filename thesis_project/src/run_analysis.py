"""P&L attribution, statistical tests, figures, and tables for the thesis.
Reads from data/processed/, writes to results/figures/ and results/tables/.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
RAW = PROJECT / "data" / "raw"
FIG_DIR = PROJECT / "results" / "figures"
TAB_DIR = PROJECT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
Q_DIV = 0.013
TRADING_DAYS = 252

VOL_LABELS = {
    "flat_bsm": "Flat BSM",
    "svi_surface": "SVI Surface",
    "rv_cc": "RV (Close-Close)",
    "rv_park": "RV (Parkinson)",
    "rv_yz": "RV (Yang-Zhang)",
}
VOL_COLORS = {
    "flat_bsm": "#2196F3",
    "svi_surface": "#E91E63",
    "rv_cc": "#4CAF50",
    "rv_park": "#FF9800",
    "rv_yz": "#9C27B0",
}
REGIME_COLORS = {
    "Low": "#4CAF50",
    "Normal": "#2196F3",
    "High": "#FF9800",
    "Crisis": "#F44336",
}
REGIME_ORDER = ["Low", "Normal", "High", "Crisis"]


# data loading

def load_all_data():
    """Load all required datasets."""
    print("loading data...")
    results = pd.read_parquet(PROCESSED / "hedging_backtest_results.parquet")
    summary = pd.read_parquet(PROCESSED / "hedging_summary_stats.parquet")
    rv = pd.read_parquet(PROCESSED / "realized_vol_daily.parquet")
    rv["date"] = pd.to_datetime(rv["date"])

    # vix
    vix = pd.read_csv(RAW / "VIX_History.csv")
    vix["date"] = pd.to_datetime(vix["DATE"], format="mixed")
    vix = vix[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})
    vix = vix[(vix["date"] >= "2019-01-01") & (vix["date"] <= "2024-12-31")]
    vix = vix.sort_values("date")

    # underlying
    und = pd.read_csv(RAW / "spx_underlying_2019_2024.csv")
    und["date"] = pd.to_datetime(und["date"])
    und = und.sort_values("date")

    print(f"  results: {len(results)} options, rv: {len(rv)} days, vix: {len(vix)} days")
    return results, summary, rv, vix, und


# pnl attribution (el karoui decomposition)

def bsm_gamma(S, K, T, r, sigma, q=Q_DIV):
    """Vectorized BSM gamma."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_delta(S, K, T, r, sigma, cp_flag, q=Q_DIV):
    """Vectorized BSM delta."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if cp_flag == "C":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def run_pnl_attribution(results):
    """El Karoui decomposition: discrete rebalancing + vol misspec + residual."""
    print("running pnl attribution...")

    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet")
    sampled_ids = set(results["optionid"].values)
    sampled_opts = opts[opts["optionid"].isin(sampled_ids)].copy()
    sampled_opts = sampled_opts.sort_values(["optionid", "date"])

    vol_inputs = {
        "flat_bsm": "flat_bsm_iv",
        "svi_surface": "svi_implied_vol",
        "rv_cc": "rv_cc",
        "rv_park": "rv_parkinson",
        "rv_yz": "rv_yangzhang",
    }

    # accumulate per-option attribution
    attrib_rows = []
    grouped = dict(list(sampled_opts.groupby("optionid")))

    for i, oid in enumerate(results["optionid"].values):
        if (i + 1) % 500 == 0:
            print(f"  attribution: option {i+1}/{len(results)}")

        odata = grouped.get(oid)
        if odata is None or len(odata) < 3:
            continue

        S = odata["spot"].values.astype(np.float64)
        K = float(odata["strike"].iloc[0])
        V = odata["mid_price"].values.astype(np.float64)
        T_arr = odata["T"].values.astype(np.float64)
        r_arr = odata["rf_rate"].values.astype(np.float64)
        cp = odata["cp_flag"].iloc[0]
        n = len(S)

        dS = np.diff(S)
        dV = np.diff(V)
        # dt in years between observations (assume ~1 trading day)
        dt = np.diff(T_arr)
        dt = np.abs(dt)  # T decreases, so diff is negative
        dt = np.maximum(dt, 1e-10)

        meta = results[results["optionid"] == oid].iloc[0]
        row = {
            "optionid": oid,
            "vix_regime": meta["vix_regime_at_entry"],
            "moneyness_bucket": meta["moneyness_bucket"],
            "maturity_bucket": meta["maturity_bucket"],
        }

        for vol_name, vol_col in vol_inputs.items():
            sigma_h = pd.Series(odata[vol_col].values, dtype=np.float64).ffill().bfill().values
            sigma_h = np.maximum(sigma_h, 0.01)

            # bsm gamma using hedge vol
            gamma = bsm_gamma(S, K, T_arr, r_arr, sigma_h)

            # realized daily variance: dS^2 / S^2
            realized_var_daily = (dS / S[:-1]) ** 2  # daily realized variance
            hedge_var_daily = sigma_h[:-1] ** 2 * dt  # hedge variance * dt

            # (a) discrete rebalancing error
            discrete_err = 0.5 * gamma[:-1] * (dS ** 2 - sigma_h[:-1] ** 2 * S[:-1] ** 2 * dt)

            # use rv_cc as realized vol proxy for vol misspec
            sigma_r = pd.Series(odata["rv_cc"].values, dtype=np.float64).ffill().bfill().values
            sigma_r = np.maximum(sigma_r, 0.01)

            # (b) vol misspecification
            vol_misspec = 0.5 * gamma[:-1] * S[:-1] ** 2 * (sigma_r[:-1] ** 2 - sigma_h[:-1] ** 2) * dt

            # total daily hedge pnl
            delta_h = bsm_delta(S, K, T_arr, r_arr, sigma_h, cp)
            total_daily = delta_h[:-1] * dS - dV

            # (c) residual
            residual = total_daily - discrete_err - vol_misspec

            row[f"discrete_err_{vol_name}"] = float(np.sum(discrete_err))
            row[f"vol_misspec_{vol_name}"] = float(np.sum(vol_misspec))
            row[f"residual_{vol_name}"] = float(np.sum(residual))
            row[f"total_pnl_{vol_name}"] = float(np.sum(total_daily))

        attrib_rows.append(row)

    attrib = pd.DataFrame(attrib_rows)
    print(f"  attribution computed for {len(attrib)} options")
    return attrib


def summarize_attribution(attrib):
    """Covariance-based variance attribution per VIX regime."""
    print("  variance attribution by regime (cov decomposition)...")

    vol_names = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]
    components = ["discrete_err", "vol_misspec", "residual"]

    rows = []
    for regime in REGIME_ORDER + ["ALL"]:
        if regime == "ALL":
            grp = attrib
        else:
            grp = attrib[attrib["vix_regime"] == regime]
        if len(grp) < 5:
            continue
        for vn in vol_names:
            total = grp[f"total_pnl_{vn}"].values
            total_var = np.var(total, ddof=1)
            if total_var < 1e-10:
                continue
            r = {"vix_regime": regime, "vol_input": vn, "n": len(grp), "total_var": total_var}
            for comp in components:
                comp_vals = grp[f"{comp}_{vn}"].values
                # cov(component, total) / var(total)
                cov_ct = np.cov(comp_vals, total, ddof=1)[0, 1]
                r[f"{comp}_pct"] = cov_ct / total_var * 100
                r[f"{comp}_mean"] = comp_vals.mean()
                r[f"{comp}_var"] = np.var(comp_vals, ddof=1)
            rows.append(r)

    attrib_summary = pd.DataFrame(rows)

    for regime in REGIME_ORDER:
        rdf = attrib_summary[attrib_summary["vix_regime"] == regime]
        if len(rdf) == 0:
            continue
        print(f"\n  {regime} regime:")
        print(f"  {'Vol Input':>15s}  {'Discrete%':>10s}  {'VolMisspec%':>12s}  {'Residual%':>10s}  {'Sum%':>6s}")
        for _, row in rdf.iterrows():
            s = row["discrete_err_pct"] + row["vol_misspec_pct"] + row["residual_pct"]
            print(f"  {row['vol_input']:>15s}  {row['discrete_err_pct']:10.1f}  "
                  f"{row['vol_misspec_pct']:12.1f}  {row['residual_pct']:10.1f}  {s:6.1f}")

    return attrib_summary


# statistical tests

def run_statistical_tests(results):
    """Paired t-tests and F-tests comparing vol inputs."""
    print("running statistical tests...")

    test_results = []

    # benchmark: flat_bsm
    bm = results["pnl_flat_bsm"].values

    for comp_col, comp_name in [("pnl_svi_surface", "SVI Surface"),
                                 ("pnl_rv_cc", "RV Close-Close")]:
        comp = results[comp_col].values
        valid = ~(np.isnan(bm) | np.isnan(comp))
        bm_v = bm[valid]
        comp_v = comp[valid]

        # paired t-test on absolute hedging errors
        abs_bm = np.abs(bm_v)
        abs_comp = np.abs(comp_v)
        t_stat, t_pval = stats.ttest_rel(abs_comp, abs_bm)

        # f-test: ratio of variances
        var_bm = np.var(bm_v, ddof=1)
        var_comp = np.var(comp_v, ddof=1)
        f_stat = var_comp / var_bm
        df1 = len(comp_v) - 1
        df2 = len(bm_v) - 1
        # two-sided f-test p-value
        f_pval = 2 * min(stats.f.cdf(f_stat, df1, df2),
                         1 - stats.f.cdf(f_stat, df1, df2))

        # diebold-mariano test on squared errors
        sq_bm = bm_v ** 2
        sq_comp = comp_v ** 2
        d = sq_comp - sq_bm
        dm_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
        dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        print(f"  {comp_name} vs Flat BSM:")
        print(f"    paired t (|error|): t={t_stat:.3f}, p={t_pval:.4f}")
        print(f"    f-test (var ratio): F={f_stat:.3f}, p={f_pval:.4f}")
        print(f"    diebold-mariano:    DM={dm_stat:.3f}, p={dm_pval:.4f}")

        test_results.append({
            "comparison": f"{comp_name} vs Flat BSM",
            "t_stat": t_stat, "t_pval": t_pval,
            "f_stat": f_stat, "f_pval": f_pval,
            "dm_stat": dm_stat, "dm_pval": dm_pval,
            "var_benchmark": var_bm, "var_comparison": var_comp,
            "mean_abs_err_benchmark": abs_bm.mean(),
            "mean_abs_err_comparison": abs_comp.mean(),
        })

    return pd.DataFrame(test_results)


# figures

def add_regime_shading(ax, vix_df):
    """Add VIX regime background shading to axis."""
    vix = vix_df.copy()
    vix = vix.sort_values("date").reset_index(drop=True)
    regime_bounds = [(0, 15, REGIME_COLORS["Low"], 0.08),
                     (15, 25, REGIME_COLORS["Normal"], 0.05),
                     (25, 35, REGIME_COLORS["High"], 0.12),
                     (35, 200, REGIME_COLORS["Crisis"], 0.18)]

    for lo, hi, color, alpha in regime_bounds:
        mask = (vix["vix"] >= lo) & (vix["vix"] < hi)
        dates = vix.loc[mask, "date"]
        if len(dates) == 0:
            continue
        # find contiguous blocks
        blocks = []
        block_start = None
        prev_idx = None
        for idx in dates.index:
            if block_start is None:
                block_start = idx
            elif idx - prev_idx > 1:
                blocks.append((vix.loc[block_start, "date"], vix.loc[prev_idx, "date"]))
                block_start = idx
            prev_idx = idx
        if block_start is not None:
            blocks.append((vix.loc[block_start, "date"], vix.loc[prev_idx, "date"]))
        for d0, d1 in blocks:
            ax.axvspan(d0, d1, alpha=alpha, color=color, linewidth=0)


def fig1_spx_vix(und, vix):
    """SPX price + VIX with regime shading."""
    print("  fig 1: spx + vix...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    add_regime_shading(ax1, vix)
    add_regime_shading(ax2, vix)

    ax1.plot(und["date"], und["close"], color="#1565C0", linewidth=0.8)
    ax1.set_ylabel("SPX Index Level", fontsize=12)
    ax1.set_title("SPX Index and VIX (2019–2024) with Volatility Regime Shading", fontsize=14)

    ax2.plot(vix["date"], vix["vix"], color="#C62828", linewidth=0.8)
    ax2.axhline(15, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.axhline(25, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.axhline(35, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("VIX", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=REGIME_COLORS[r], alpha=0.3, label=f"{r}")
                       for r in REGIME_ORDER]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9, title="VIX Regime")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_spx_vix_regimes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig2_svi_rmse(vix):
    """SVI calibration RMSE time series."""
    print("  fig 2: svi rmse...")

    # compare svi_implied_vol to impl_volatility per (date, exdate)
    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet",
                           columns=["date", "exdate", "impl_volatility", "svi_implied_vol"])

    diff = opts["svi_implied_vol"] - opts["impl_volatility"]
    opts["sq_diff"] = diff ** 2
    daily_rmse = opts.groupby("date")["sq_diff"].mean().reset_index()
    daily_rmse["rmse_bps"] = np.sqrt(daily_rmse["sq_diff"]) * 10_000
    daily_rmse["date"] = pd.to_datetime(daily_rmse["date"])
    daily_rmse = daily_rmse.sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))
    add_regime_shading(ax, vix)
    ax.plot(daily_rmse["date"], daily_rmse["rmse_bps"], color="#1565C0", linewidth=0.6, alpha=0.8)
    # rolling 21-day average
    daily_rmse["rmse_ma"] = daily_rmse["rmse_bps"].rolling(21, min_periods=5).mean()
    ax.plot(daily_rmse["date"], daily_rmse["rmse_ma"], color="#C62828", linewidth=1.5, label="21-day MA")
    ax.set_ylabel("SVI Fit RMSE (bps)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("SVI Calibration RMSE Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_svi_rmse_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig3_svi_examples():
    """Example SVI fits, one panel per VIX regime."""
    print("  fig 3: svi fit examples...")

    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet")
    opts["date"] = pd.to_datetime(opts["date"])

    # one representative date per regime
    regime_dates = {
        "Low": "2024-01-15",      # VIX ~13
        "Normal": "2023-06-15",   # VIX ~15-20
        "High": "2022-06-15",     # VIX ~25-30
        "Crisis": "2020-03-16",   # VIX ~80
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (regime, target_date) in zip(axes.flatten(), regime_dates.items()):
        target = pd.Timestamp(target_date)
        # find closest available date
        available = opts["date"].unique()
        closest = min(available, key=lambda d: abs(d - target))

        day_data = opts[opts["date"] == closest]
        if len(day_data) == 0:
            ax.set_title(f"{regime}: No data")
            continue

        # pick exdate with most strikes
        exdate_counts = day_data.groupby("exdate").size()
        best_ex = exdate_counts.idxmax()
        slice_data = day_data[day_data["exdate"] == best_ex].sort_values("strike")

        vix_val = slice_data["vix_close"].iloc[0]
        dte = slice_data["dte"].iloc[0]

        ax.scatter(slice_data["moneyness"], slice_data["impl_volatility"],
                   s=20, color="#1565C0", alpha=0.7, label="Market IV", zorder=3)
        ax.plot(slice_data["moneyness"].values, slice_data["svi_implied_vol"].values,
                color="#C62828", linewidth=1.5, label="SVI Fit", zorder=2)

        ax.set_xlabel("Moneyness (K/S)", fontsize=10)
        ax.set_ylabel("Implied Volatility", fontsize=10)
        ax.set_title(f"{regime} Regime — {closest.strftime('%Y-%m-%d')} "
                     f"(VIX={vix_val:.0f}, DTE={dte})", fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.suptitle("SVI Smile Calibration Examples Across VIX Regimes", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_svi_fit_examples.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig4_hedging_error_hist(results):
    """Hedging error distributions, overlaid histograms."""
    print("  fig 4: hedging error histograms...")

    fig, ax = plt.subplots(figsize=(12, 6))

    pnl_cols = ["pnl_flat_bsm", "pnl_svi_surface", "pnl_rv_cc", "pnl_rv_park", "pnl_rv_yz"]
    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]

    # common bins
    all_pnl = pd.concat([results[c] for c in pnl_cols]).dropna()
    p1, p99 = all_pnl.quantile(0.01), all_pnl.quantile(0.99)
    bins = np.linspace(p1, p99, 60)

    for col, vk in zip(pnl_cols, vol_keys):
        vals = results[col].dropna()
        ax.hist(vals, bins=bins, alpha=0.35, color=VOL_COLORS[vk],
                label=f"{VOL_LABELS[vk]} (std=${vals.std():.1f})", density=True)

    ax.set_xlabel("Hedging Error ($)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Hedging Errors by Volatility Input", fontsize=14)
    ax.legend(fontsize=10)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_hedging_error_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig5_hedging_std_by_regime(results):
    """Hedging error std by VIX regime, grouped bar chart."""
    print("  fig 5: hedging std by regime...")

    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]
    pnl_cols = [f"pnl_{v}" for v in vol_keys]

    # std per regime
    regime_std = results.groupby("vix_regime_at_entry", observed=True)[pnl_cols].std()
    regime_std = regime_std.reindex(REGIME_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(REGIME_ORDER))
    width = 0.15

    for i, vk in enumerate(vol_keys):
        col = f"pnl_{vk}"
        vals = regime_std[col].values
        bars = ax.bar(x + i * width, vals, width, label=VOL_LABELS[vk],
                      color=VOL_COLORS[vk], alpha=0.85)

    ax.set_xlabel("VIX Regime at Entry", fontsize=12)
    ax.set_ylabel("Std of Hedging Error ($)", fontsize=12)
    ax.set_title("Hedging Error Volatility by VIX Regime and Vol Input", fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(REGIME_ORDER, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_hedging_std_by_regime.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig6_pnl_attribution(attrib):
    """P&L attribution stacked bars by VIX regime (cov decomposition)."""
    print("  fig 6: pnl attribution by regime...")

    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]
    components = ["discrete_err", "vol_misspec", "residual"]
    comp_colors = {"discrete_err": "#2196F3", "vol_misspec": "#FF9800", "residual": "#9E9E9E"}
    comp_labels = {"discrete_err": "Discrete Rebalancing", "vol_misspec": "Vol Misspecification",
                   "residual": "Higher-Order Residual"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

    for ax, regime in zip(axes, REGIME_ORDER):
        grp = attrib[attrib["vix_regime"] == regime]
        if len(grp) < 5:
            ax.set_title(f"{regime}: insufficient data")
            continue

        x = np.arange(len(vol_keys))
        width = 0.6

        # cov-based attribution: cov(comp, total) / var(total)
        for comp in components:
            pcts = np.zeros(len(vol_keys))
            for j, vk in enumerate(vol_keys):
                total = grp[f"total_pnl_{vk}"].values
                comp_vals = grp[f"{comp}_{vk}"].values
                total_var = np.var(total, ddof=1)
                if total_var > 1e-10:
                    cov_ct = np.cov(comp_vals, total, ddof=1)[0, 1]
                    pcts[j] = cov_ct / total_var * 100

            # split positive and negative contributions
            pos = np.maximum(pcts, 0)
            neg = np.minimum(pcts, 0)

            if comp == components[0]:
                bottoms_pos = np.zeros(len(vol_keys))
                bottoms_neg = np.zeros(len(vol_keys))

            ax.bar(x, pos, width, bottom=bottoms_pos, label=comp_labels[comp],
                   color=comp_colors[comp], alpha=0.85)
            ax.bar(x, neg, width, bottom=bottoms_neg, color=comp_colors[comp], alpha=0.85)
            bottoms_pos += pos
            bottoms_neg += neg

        ax.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(f"{regime}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([VOL_LABELS[v].replace(" ", "\n") for v in vol_keys],
                           fontsize=7, rotation=0)
        if regime == "Low":
            ax.set_ylabel("Variance Contribution (%)", fontsize=11)

    axes[0].legend(fontsize=8, loc="upper left")
    plt.suptitle("P&L Variance Attribution by VIX Regime (Covariance Decomposition)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_pnl_attribution_by_regime.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig7_rv_vs_iv(rv, vix):
    """Realized vol estimators vs ATM implied vol time series."""
    print("  fig 7: rv vs iv...")

    # atm iv from short-dated options
    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet",
                           columns=["date", "flat_bsm_iv", "dte"])
    short = opts[(opts["dte"] >= 25) & (opts["dte"] <= 35)]
    atm_iv = short.groupby("date")["flat_bsm_iv"].median().reset_index()
    atm_iv.columns = ["date", "atm_iv"]
    atm_iv["date"] = pd.to_datetime(atm_iv["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    add_regime_shading(ax, vix)

    ax.plot(rv["date"], rv["rv_cc"], linewidth=0.8, alpha=0.8,
            color=VOL_COLORS["rv_cc"], label="RV (Close-Close)")
    ax.plot(rv["date"], rv["rv_parkinson"], linewidth=0.8, alpha=0.8,
            color=VOL_COLORS["rv_park"], label="RV (Parkinson)")
    ax.plot(rv["date"], rv["rv_yangzhang"], linewidth=0.8, alpha=0.8,
            color=VOL_COLORS["rv_yz"], label="RV (Yang-Zhang)")
    ax.plot(atm_iv["date"], atm_iv["atm_iv"], linewidth=1.2,
            color=VOL_COLORS["flat_bsm"], label="ATM IV (30-day)")

    ax.set_ylabel("Annualized Volatility", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Realized Volatility Estimators vs ATM Implied Volatility (2019–2024)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_rv_vs_iv_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig8_svi_rmse_vs_hedge_error(results):
    """Scatter: SVI calibration quality vs hedging error."""
    print("  fig 8: svi rmse vs hedge error...")

    # per-option svi fit quality
    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet",
                           columns=["optionid", "impl_volatility", "svi_implied_vol", "vix_regime"])
    sampled = opts[opts["optionid"].isin(results["optionid"].values)].copy()
    sampled["abs_diff_bps"] = (sampled["svi_implied_vol"] - sampled["impl_volatility"]).abs() * 10_000

    svi_quality = sampled.groupby("optionid").agg(
        mean_abs_diff_bps=("abs_diff_bps", "mean"),
        regime=("vix_regime", "first"),
    ).reset_index()

    merged = results[["optionid", "pnl_svi_surface", "vix_regime_at_entry"]].merge(
        svi_quality, on="optionid", how="left",
    )
    merged["abs_hedge_err"] = merged["pnl_svi_surface"].abs()

    fig, ax = plt.subplots(figsize=(10, 7))

    for regime in REGIME_ORDER:
        mask = merged["vix_regime_at_entry"] == regime
        grp = merged[mask]
        ax.scatter(grp["mean_abs_diff_bps"], grp["abs_hedge_err"],
                   s=15, alpha=0.5, color=REGIME_COLORS[regime], label=regime)

    # regression line
    x = merged["mean_abs_diff_bps"].values
    y = merged["abs_hedge_err"].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 10:
        slope, intercept, r_val, p_val, _ = stats.linregress(x[valid], y[valid])
        x_line = np.linspace(np.nanmin(x), np.nanpercentile(x, 99), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5,
                label=f"OLS: R²={r_val**2:.3f}, p={p_val:.3f}")

    ax.set_xlabel("Mean |SVI IV − Market IV| (bps)", fontsize=12)
    ax.set_ylabel("|Hedging Error| ($)", fontsize=12)
    ax.set_title("SVI Calibration Quality vs Hedging Error", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_svi_rmse_vs_hedge_error.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# tables

def table1_dataset_summary(results):
    """Dataset summary: options per regime/moneyness/maturity."""
    print("  table 1: dataset summary...")

    rows = []

    # by vix regime
    for regime in REGIME_ORDER:
        n = (results["vix_regime_at_entry"] == regime).sum()
        rows.append({"category": "VIX Regime", "group": regime, "n_options": n})

    # by moneyness
    for m in results["moneyness_bucket"].value_counts().sort_index().index:
        n = (results["moneyness_bucket"] == m).sum()
        rows.append({"category": "Moneyness", "group": m, "n_options": n})

    # by maturity
    for m in ["short", "medium", "long"]:
        n = (results["maturity_bucket"] == m).sum()
        rows.append({"category": "Maturity", "group": m, "n_options": n})

    # by option type
    for cp in ["C", "P"]:
        n = (results["cp_flag"] == cp).sum()
        rows.append({"category": "Type", "group": "Call" if cp == "C" else "Put", "n_options": n})

    rows.append({"category": "Total", "group": "All", "n_options": len(results)})

    # date range
    rows.append({"category": "Date Range", "group": f"{results['entry_date'].min().date()} to {results['entry_date'].max().date()}", "n_options": len(results)})

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table1_dataset_summary.csv", index=False)
    return df


def table2_svi_quality(results):
    """SVI calibration quality by VIX regime."""
    print("  table 2: svi quality...")

    opts = pd.read_parquet(PROCESSED / "spx_with_svi.parquet",
                           columns=["optionid", "impl_volatility", "svi_implied_vol",
                                    "vix_regime", "date", "exdate"])

    sampled = opts[opts["optionid"].isin(results["optionid"].values)].copy()
    sampled["diff_bps"] = (sampled["svi_implied_vol"] - sampled["impl_volatility"]) * 10_000
    sampled["abs_diff_bps"] = sampled["diff_bps"].abs()

    rows = []
    for regime in REGIME_ORDER + ["ALL"]:
        if regime == "ALL":
            grp = sampled
        else:
            grp = sampled[sampled["vix_regime"] == regime]
        if len(grp) == 0:
            continue
        rows.append({
            "vix_regime": regime,
            "n_obs": len(grp),
            "mean_diff_bps": grp["diff_bps"].mean(),
            "std_diff_bps": grp["diff_bps"].std(),
            "mae_bps": grp["abs_diff_bps"].mean(),
            "rmse_bps": np.sqrt((grp["diff_bps"] ** 2).mean()),
            "median_abs_bps": grp["abs_diff_bps"].median(),
            "p95_abs_bps": grp["abs_diff_bps"].quantile(0.95),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table2_svi_quality_by_regime.csv", index=False)
    return df


def table3_hedging_comparison(results):
    """Main hedging error comparison table + regime breakdown."""
    print("  table 3: hedging comparison...")

    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]
    pnl_cols = [f"pnl_{v}" for v in vol_keys]

    rows = []
    bm_std = results["pnl_flat_bsm"].std()

    # overall
    for vk in vol_keys:
        vals = results[f"pnl_{vk}"].dropna()
        rows.append({
            "vol_input": VOL_LABELS[vk],
            "regime": "ALL",
            "n": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "median": vals.median(),
            "mae": vals.abs().mean(),
            "rmse": np.sqrt((vals ** 2).mean()),
            "gain_pct": (1 - vals.std() / bm_std) * 100,
            "pct_positive": (vals > 0).mean() * 100,
        })

    # by regime
    for regime in REGIME_ORDER:
        grp = results[results["vix_regime_at_entry"] == regime]
        bm_r = grp["pnl_flat_bsm"].std()
        for vk in vol_keys:
            vals = grp[f"pnl_{vk}"].dropna()
            if len(vals) < 5:
                continue
            rows.append({
                "vol_input": VOL_LABELS[vk],
                "regime": regime,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "mae": vals.abs().mean(),
                "rmse": np.sqrt((vals ** 2).mean()),
                "gain_pct": (1 - vals.std() / bm_r) * 100,
                "pct_positive": (vals > 0).mean() * 100,
            })

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table3_hedging_error_comparison.csv", index=False)
    return df


def table4_attribution(attrib):
    """P&L attribution by regime (cov decomposition)."""
    print("  table 4: pnl attribution...")

    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]
    components = ["discrete_err", "vol_misspec", "residual"]

    rows = []
    for regime in REGIME_ORDER + ["ALL"]:
        if regime == "ALL":
            grp = attrib
        else:
            grp = attrib[attrib["vix_regime"] == regime]
        if len(grp) < 5:
            continue
        for vk in vol_keys:
            total = grp[f"total_pnl_{vk}"].values
            total_var = np.var(total, ddof=1)
            r = {"regime": regime, "vol_input": VOL_LABELS[vk], "n": len(grp),
                 "total_var": total_var}
            for comp in components:
                comp_vals = grp[f"{comp}_{vk}"].values
                cm = comp_vals.mean()
                cv = np.var(comp_vals, ddof=1)
                cov_ct = np.cov(comp_vals, total, ddof=1)[0, 1] if total_var > 0 else 0
                r[f"{comp}_mean"] = cm
                r[f"{comp}_var"] = cv
                r[f"{comp}_cov_pct"] = cov_ct / total_var * 100 if total_var > 0 else 0
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table4_pnl_attribution.csv", index=False)
    return df


def table5_best_vol(results):
    """Best vol input per moneyness x maturity cell."""
    print("  table 5: best vol per cell...")

    vol_keys = ["flat_bsm", "svi_surface", "rv_cc", "rv_park", "rv_yz"]

    rows = []
    for mb in sorted(results["moneyness_bucket"].unique()):
        for mtb in ["short", "medium", "long"]:
            grp = results[(results["moneyness_bucket"] == mb) &
                          (results["maturity_bucket"] == mtb)]
            if len(grp) < 5:
                continue

            best_vk = None
            best_rmse = np.inf
            cell_stats = {"moneyness": mb, "maturity": mtb, "n": len(grp)}

            for vk in vol_keys:
                vals = grp[f"pnl_{vk}"].dropna()
                rmse = np.sqrt((vals ** 2).mean())
                cell_stats[f"rmse_{vk}"] = rmse
                cell_stats[f"std_{vk}"] = vals.std()
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_vk = vk

            cell_stats["best_vol"] = VOL_LABELS.get(best_vk, best_vk)
            cell_stats["best_rmse"] = best_rmse
            rows.append(cell_stats)

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table5_best_vol_per_cell.csv", index=False)
    return df


# main

if __name__ == "__main__":
    import time
    t0 = time.time()

    results, summary, rv, vix, und = load_all_data()

    # pnl attribution
    attrib = run_pnl_attribution(results)
    attrib_summary = summarize_attribution(attrib)

    # statistical tests
    test_results = run_statistical_tests(results)
    test_results.to_csv(TAB_DIR / "statistical_tests.csv", index=False)

    # figures
    print("generating figures...")
    fig1_spx_vix(und, vix)
    fig2_svi_rmse(vix)
    fig3_svi_examples()
    fig4_hedging_error_hist(results)
    fig5_hedging_std_by_regime(results)
    fig6_pnl_attribution(attrib)
    fig7_rv_vs_iv(rv, vix)
    fig8_svi_rmse_vs_hedge_error(results)

    # tables
    print("generating tables...")
    t1 = table1_dataset_summary(results)
    t2 = table2_svi_quality(results)
    t3 = table3_hedging_comparison(results)
    t4 = table4_attribution(attrib)
    t5 = table5_best_vol(results)

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s -- figures: {FIG_DIR}, tables: {TAB_DIR}")

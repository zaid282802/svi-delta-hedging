"""SVI volatility surface calibration and arbitrage checks."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from src.config import (
    IV_LOWER_BOUND,
    IV_UPPER_BOUND,
    RANDOM_SEED,
)


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """SVI raw parameterization for total variance."""
    k = np.asarray(k, dtype=np.float64)
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


def svi_implied_vol(
    k: np.ndarray,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Convert SVI total variance to implied vol."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0.0) / T)


def calibrate_svi_slice(
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    forward: float,
    T: float,
) -> Dict:
    """Calibrate SVI params to one expiration slice using DE + L-BFGS-B."""
    strikes = np.asarray(strikes, dtype=np.float64)
    market_ivs = np.asarray(market_ivs, dtype=np.float64)

    k = np.log(strikes / forward)
    market_w = market_ivs ** 2 * T

    bounds = [
        (-0.5, 0.5),      # a
        (0.001, 2.0),     # b
        (-0.99, 0.99),    # rho
        (-0.5, 0.5),      # m
        (0.001, 2.0),     # sigma
    ]

    def objective(params):
        a, b, rho, m, sigma = params
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        return np.sum((w_model - market_w) ** 2)

    def _calibrate_with_seed(seed: int) -> Tuple[np.ndarray, float]:
        result_de = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=300,
            seed=seed,
            tol=1e-12,
            polish=False,
        )
        result_local = minimize(
            objective,
            x0=result_de.x,
            method="L-BFGS-B",
            bounds=bounds,
        )
        return result_local.x, result_local.fun

    best_params, best_fun = _calibrate_with_seed(RANDOM_SEED)

    w_fit = svi_total_variance(k, *best_params)
    iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
    rmse_bps = np.sqrt(np.mean((iv_fit - market_ivs) ** 2)) * 10_000

    # Retry with different seeds if fit is poor
    if rmse_bps > 200.0:
        for seed in [RANDOM_SEED + 1, RANDOM_SEED + 2, RANDOM_SEED + 3]:
            params_i, fun_i = _calibrate_with_seed(seed)
            if fun_i < best_fun:
                best_params = params_i
                best_fun = fun_i

        w_fit = svi_total_variance(k, *best_params)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
        rmse_bps = np.sqrt(np.mean((iv_fit - market_ivs) ** 2)) * 10_000

    max_error_bps = np.max(np.abs(iv_fit - market_ivs)) * 10_000
    ss_res = np.sum((market_w - w_fit) ** 2)
    ss_tot = np.sum((market_w - np.mean(market_w)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    a, b, rho, m, sigma = best_params

    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma),
        "rmse_bps": float(rmse_bps),
        "max_error_bps": float(max_error_bps),
        "r_squared": float(r_squared),
        "T": float(T),
        "forward": float(forward),
        "n_points": len(strikes),
    }


def check_butterfly_arbitrage(
    k_grid: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Tuple[np.ndarray, bool]:
    """Check butterfly no-arbitrage condition on an SVI slice."""
    k_grid = np.asarray(k_grid, dtype=np.float64)
    dk = k_grid[1] - k_grid[0] if len(k_grid) > 1 else 0.01

    w = svi_total_variance(k_grid, a, b, rho, m, sigma)

    w_plus = svi_total_variance(k_grid + dk, a, b, rho, m, sigma)
    w_minus = svi_total_variance(k_grid - dk, a, b, rho, m, sigma)
    w_prime = (w_plus - w_minus) / (2.0 * dk)

    w_double_prime = (w_plus - 2.0 * w + w_minus) / (dk ** 2)

    w_safe = np.maximum(w, 1e-12)

    g = (
        (1.0 - k_grid * w_prime / (2.0 * w_safe)) ** 2
        - (w_prime ** 2) / 4.0 * (1.0 / w_safe + 0.25)
        + w_double_prime / 2.0
    )

    is_arb_free = bool(np.all(g >= -1e-10))
    return g, is_arb_free


def check_calendar_arbitrage(surface_params: Dict[float, Dict]) -> Dict:
    """Check calendar spread arbitrage across expirations."""
    sorted_T = sorted(surface_params.keys())
    k_test = np.linspace(-0.5, 0.5, 101)
    violations: List[Dict] = []

    for i in range(len(sorted_T) - 1):
        T1 = sorted_T[i]
        T2 = sorted_T[i + 1]
        p1 = surface_params[T1]
        p2 = surface_params[T2]

        w1 = svi_total_variance(k_test, p1["a"], p1["b"], p1["rho"], p1["m"], p1["sigma"])
        w2 = svi_total_variance(k_test, p2["a"], p2["b"], p2["rho"], p2["m"], p2["sigma"])

        diff = w1 - w2
        violation_mask = diff > 1e-10

        if np.any(violation_mask):
            violations.append({
                "T1": T1,
                "T2": T2,
                "max_violation": float(np.max(diff[violation_mask])),
                "n_violations": int(np.sum(violation_mask)),
                "k_range": (
                    float(k_test[violation_mask][0]),
                    float(k_test[violation_mask][-1]),
                ),
            })

    return {
        "is_arb_free": len(violations) == 0,
        "violations": violations,
    }


def build_vol_surface(
    data: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
) -> Dict:
    """Build full SVI vol surface from options data."""
    surface_params: Dict[float, Dict] = {}

    if "expiration" in data.columns:
        grouped = data.groupby("expiration")
    else:
        grouped = data.groupby("T")

    summary_rows = []

    for group_key, group_df in grouped:
        T = group_df["T"].iloc[0]

        if len(group_df) < 5:
            continue

        forward = spot * np.exp((r - q) * T)

        # Use OTM options: puts below forward, calls above
        otm_mask = (
            ((group_df["option_type"] == "put") & (group_df["strike"] <= forward))
            | ((group_df["option_type"] == "call") & (group_df["strike"] > forward))
        )
        otm_df = group_df[otm_mask]

        if len(otm_df) < 5:
            otm_df = group_df

        strikes = otm_df["strike"].values
        ivs = otm_df["computed_iv"].values

        params = calibrate_svi_slice(strikes, ivs, forward, T)

        k_test = np.linspace(-0.5, 0.5, 201)
        _, bf_arb_free = check_butterfly_arbitrage(
            k_test, params["a"], params["b"], params["rho"], params["m"], params["sigma"]
        )
        params["butterfly_arb_free"] = bf_arb_free

        surface_params[T] = params

        summary_rows.append({
            "T": f"{T:.4f}",
            "N_pts": params["n_points"],
            "RMSE_bps": f"{params['rmse_bps']:.1f}",
            "MaxErr_bps": f"{params['max_error_bps']:.1f}",
            "R2": f"{params['r_squared']:.6f}",
            "BF_arb_free": bf_arb_free,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n=== SVI Calibration Summary ===")
        print(summary_df.to_string(index=False))
        print(f"\nTotal slices calibrated: {len(summary_rows)}")

    return surface_params


def interpolate_surface(
    surface_params: Dict[float, Dict],
    strike_grid: np.ndarray,
    T_grid: np.ndarray,
    spot: float,
    r: float,
    q: float,
) -> np.ndarray:
    """Interpolate IV across the calibrated surface."""
    sorted_T = np.array(sorted(surface_params.keys()))
    iv_surface = np.full((len(T_grid), len(strike_grid)), np.nan)

    for i, T in enumerate(T_grid):
        forward = spot * np.exp((r - q) * T)
        k = np.log(strike_grid / forward)

        if T <= sorted_T[0]:
            p = surface_params[sorted_T[0]]
            iv_surface[i, :] = svi_implied_vol(k, T if T > 0 else sorted_T[0],
                                                p["a"], p["b"], p["rho"], p["m"], p["sigma"])
        elif T >= sorted_T[-1]:
            p = surface_params[sorted_T[-1]]
            iv_surface[i, :] = svi_implied_vol(k, T,
                                                p["a"], p["b"], p["rho"], p["m"], p["sigma"])
        else:
            idx = np.searchsorted(sorted_T, T) - 1
            T1 = sorted_T[idx]
            T2 = sorted_T[idx + 1]
            p1 = surface_params[T1]
            p2 = surface_params[T2]

            f1 = spot * np.exp((r - q) * T1)
            f2 = spot * np.exp((r - q) * T2)

            k1 = np.log(strike_grid / f1)
            k2 = np.log(strike_grid / f2)

            w1 = svi_total_variance(k1, p1["a"], p1["b"], p1["rho"], p1["m"], p1["sigma"])
            w2 = svi_total_variance(k2, p2["a"], p2["b"], p2["rho"], p2["m"], p2["sigma"])

            alpha = (T - T1) / (T2 - T1)
            w_interp = (1.0 - alpha) * w1 + alpha * w2

            iv_surface[i, :] = np.sqrt(np.maximum(w_interp, 0.0) / T)

    return iv_surface


def delta_to_strike(
    delta: float,
    S: float,
    T: float,
    r: float,
    q: float,
    sigma_atm: float,
) -> float:
    """Convert BS delta to strike price."""
    sqrt_T = np.sqrt(T)
    d1 = norm.ppf(delta * np.exp(q * T))
    K = S * np.exp(-d1 * sigma_atm * sqrt_T + (r - q + 0.5 * sigma_atm ** 2) * T)
    return float(K)


def compute_smile_metrics(
    surface_params_slice: Dict,
    forward: float,
    T: float,
) -> Dict:
    """Compute smile metrics (ATM IV, risk reversal, butterfly) for a slice."""
    p = surface_params_slice

    atm_iv = float(svi_implied_vol(np.array([0.0]), T,
                                    p["a"], p["b"], p["rho"], p["m"], p["sigma"])[0])

    sqrt_T = np.sqrt(T)

    d1_25c = norm.ppf(0.25)
    k_25c = -d1_25c * atm_iv * sqrt_T + 0.5 * atm_iv ** 2 * T

    d1_25p = norm.ppf(0.75)
    k_25p = -d1_25p * atm_iv * sqrt_T + 0.5 * atm_iv ** 2 * T

    iv_25c = float(svi_implied_vol(np.array([k_25c]), T,
                                    p["a"], p["b"], p["rho"], p["m"], p["sigma"])[0])
    iv_25p = float(svi_implied_vol(np.array([k_25p]), T,
                                    p["a"], p["b"], p["rho"], p["m"], p["sigma"])[0])

    rr_25d = iv_25c - iv_25p
    bf_25d = 0.5 * (iv_25c + iv_25p) - atm_iv

    dk = 0.01
    iv_plus = float(svi_implied_vol(np.array([dk]), T,
                                     p["a"], p["b"], p["rho"], p["m"], p["sigma"])[0])
    iv_minus = float(svi_implied_vol(np.array([-dk]), T,
                                      p["a"], p["b"], p["rho"], p["m"], p["sigma"])[0])
    skew_slope = (iv_plus - iv_minus) / (2.0 * dk)

    return {
        "atm_iv": atm_iv,
        "rr_25d": rr_25d,
        "bf_25d": bf_25d,
        "skew_slope": skew_slope,
    }

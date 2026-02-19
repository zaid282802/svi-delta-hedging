"""Enhanced hedging: delta-gamma hedging, P&L attribution, vega hedging simulation."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HedgeResult:
    """Container for hedge simulation results."""
    final_pnl: float
    pnl_path: np.ndarray
    gamma_pnl: np.ndarray
    theta_pnl: np.ndarray
    vega_pnl: np.ndarray
    discrete_error: np.ndarray
    total_transaction_costs: float
    spot_path: np.ndarray


def black_scholes_greeks(S, K, T, r, q, sigma):
    """All BS Greeks for a European call. S can be scalar or array."""
    if np.isscalar(sigma) and sigma <= 0:
        raise ValueError(f"Volatility must be positive, got {sigma}")
    if np.isscalar(T) and T < 0:
        raise ValueError(f"Time to expiry must be non-negative, got {T}")

    S = np.asarray(S, dtype=float)
    scalar_input = S.ndim == 0
    S = np.atleast_1d(S)

    if np.isscalar(T) and T == 0:
        intrinsic = np.maximum(S - K, 0)
        delta = np.where(S > K, 1.0, 0.0)
        zeros = np.zeros_like(S)
        result = {
            'price': intrinsic, 'delta': delta, 'gamma': zeros,
            'vega': zeros, 'theta': zeros, 'vanna': zeros, 'volga': zeros,
        }
        if scalar_input:
            return {k: v.item() for k, v in result.items()}
        return result

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Dq = np.exp(-q * T)
    Dr = np.exp(-r * T)
    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    price = S * Dq * Phi_d1 - K * Dr * Phi_d2
    delta = Dq * Phi_d1
    gamma = Dq * phi_d1 / (S * sigma * sqrt_T)
    vega = S * Dq * phi_d1 * sqrt_T / 100
    theta = (-(S * Dq * phi_d1 * sigma) / (2 * sqrt_T)
             - r * K * Dr * Phi_d2
             + q * S * Dq * Phi_d1) / 365
    vanna = -Dq * phi_d1 * d2 / sigma
    volga = S * Dq * phi_d1 * sqrt_T * d1 * d2 / sigma

    result = {
        'price': price, 'delta': delta, 'gamma': gamma,
        'vega': vega, 'theta': theta, 'vanna': vanna, 'volga': volga,
    }
    if scalar_input:
        return {k: v.item() for k, v in result.items()}
    return result


def _generate_paths(S0, r, q, sigma, T, n_steps, n_paths, seed=None):
    """Generate GBM paths as (n_paths, n_steps+1) array."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    np.cumsum(log_increments, axis=1, out=log_S[:, 1:])
    log_S[:, 1:] += np.log(S0)
    return np.exp(log_S)


def delta_gamma_hedge_simulation(
    S0, K1, K2, T, r, q, sigma_true, sigma_hedge,
    n_steps=252, n_paths=10000, transaction_cost_bps=5.0, seed=42,
):
    """Compare delta-only vs delta-gamma hedging. Vectorized across paths."""
    dt = T / n_steps
    tc = transaction_cost_bps / 10000

    S = _generate_paths(S0, r, q, sigma_true, T, n_steps, n_paths, seed)

    g0 = black_scholes_greeks(S0, K1, T, r, q, sigma_hedge)
    premium = g0['price']
    delta_0 = g0['delta']

    # --- delta-only (matches existing delta_hedge.py convention) ---
    shares_d = np.full(n_paths, delta_0)
    cash_d = np.full(n_paths, premium - delta_0 * S0)

    for i in range(1, n_steps):
        cash_d *= np.exp(r * dt)
        tau = T - i * dt
        if tau > 1e-6:
            new_delta = black_scholes_greeks(S[:, i], K1, tau, r, q, sigma_hedge)['delta']
        else:
            new_delta = np.where(S[:, i] > K1, 1.0, 0.0)
        trade = new_delta - shares_d
        cash_d -= trade * S[:, i] + np.abs(trade) * S[:, i] * tc
        shares_d = new_delta

    cash_d *= np.exp(r * dt)
    payoff1 = np.maximum(S[:, -1] - K1, 0)
    delta_only_pnl = shares_d * S[:, -1] + cash_d - payoff1

    # --- delta-gamma (same convention, adds hedge option at K2) ---
    g0_2 = black_scholes_greeks(S0, K2, T, r, q, sigma_hedge)
    opts2_0 = g0['gamma'] / g0_2['gamma'] if abs(g0_2['gamma']) > 1e-10 else 0.0
    shares_dg_0 = delta_0 - opts2_0 * g0_2['delta']

    shares_dg = np.full(n_paths, shares_dg_0)
    opts2 = np.full(n_paths, opts2_0)
    cash_dg = np.full(n_paths, premium - shares_dg_0 * S0 - opts2_0 * g0_2['price'])

    for i in range(1, n_steps):
        cash_dg *= np.exp(r * dt)
        tau = T - i * dt
        if tau > 1e-6:
            g1 = black_scholes_greeks(S[:, i], K1, tau, r, q, sigma_hedge)
            g2 = black_scholes_greeks(S[:, i], K2, tau, r, q, sigma_hedge)
            with np.errstate(divide='ignore', invalid='ignore'):
                new_opts2 = np.where(np.abs(g2['gamma']) > 1e-10,
                                     g1['gamma'] / g2['gamma'], 0.0)
            new_shares = g1['delta'] - new_opts2 * g2['delta']
            opt2_price = g2['price']
        else:
            new_opts2 = np.zeros(n_paths)
            new_shares = np.where(S[:, i] > K1, 1.0, 0.0)
            opt2_price = np.maximum(S[:, i] - K2, 0)

        o2_trade = new_opts2 - opts2
        s_trade = new_shares - shares_dg
        cash_dg -= (s_trade * S[:, i] + np.abs(s_trade) * S[:, i] * tc
                    + o2_trade * opt2_price + np.abs(o2_trade) * opt2_price * tc)
        shares_dg = new_shares
        opts2 = new_opts2

    cash_dg *= np.exp(r * dt)
    payoff2 = np.maximum(S[:, -1] - K2, 0)
    delta_gamma_pnl = shares_dg * S[:, -1] + opts2 * payoff2 + cash_dg - payoff1

    std_d = delta_only_pnl.std()
    std_dg = delta_gamma_pnl.std()
    logger.info(f"Delta-only:  mean={delta_only_pnl.mean():.4f}, std={std_d:.4f}")
    logger.info(f"Delta-gamma: mean={delta_gamma_pnl.mean():.4f}, std={std_dg:.4f}")

    return {
        'delta_only_pnl': delta_only_pnl,
        'delta_gamma_pnl': delta_gamma_pnl,
        'std_reduction_pct': (1 - std_dg / std_d) * 100 if std_d > 0 else 0.0,
    }


def pnl_attribution_analysis(
    S0, K, T, r, q, sigma_true, sigma_hedge,
    n_steps=252, n_paths=10000, seed=42,
):
    """Decompose hedge P&L variance into discrete-rebalancing, vol-misspec, and higher-order.

    Components (per-step, summed across time):
      1. disc_hedge:  0.5 * Gamma * (dS^2 - sigma_true^2 * S^2 * dt)
      2. vol_misspec: 0.5 * Gamma * S^2 * (sigma_true^2 - sigma_hedge^2) * dt
    Variance percentages use Var(component)/Var(total) to show relative magnitude.
    """
    dt = T / n_steps

    S = _generate_paths(S0, r, q, sigma_true, T, n_steps, n_paths, seed)

    g0 = black_scholes_greeks(S0, K, T, r, q, sigma_hedge)
    premium, delta_0 = g0['price'], g0['delta']
    gamma_0 = g0['gamma']

    shares = np.full(n_paths, delta_0)
    cash = np.full(n_paths, premium - delta_0 * S0)
    disc_hedge = np.zeros(n_paths)
    vol_misspec = np.zeros(n_paths)

    # Step 0 -> 1
    dS0 = S[:, 1] - S[:, 0]
    disc_hedge += 0.5 * gamma_0 * (dS0**2 - sigma_true**2 * S[:, 0]**2 * dt)
    vol_misspec += 0.5 * gamma_0 * S[:, 0]**2 * (sigma_true**2 - sigma_hedge**2) * dt

    for i in range(1, n_steps):
        cash *= np.exp(r * dt)
        tau = T - i * dt
        if tau > 1e-6:
            g = black_scholes_greeks(S[:, i], K, tau, r, q, sigma_hedge)
            delta, gamma = g['delta'], g['gamma']
        else:
            delta = np.where(S[:, i] > K, 1.0, 0.0)
            gamma = np.zeros(n_paths)

        dS = S[:, i + 1] - S[:, i]
        disc_hedge += 0.5 * gamma * (dS**2 - sigma_true**2 * S[:, i]**2 * dt)
        vol_misspec += 0.5 * gamma * S[:, i]**2 * (sigma_true**2 - sigma_hedge**2) * dt

        trade = delta - shares
        cash -= trade * S[:, i]
        shares = delta

    cash *= np.exp(r * dt)
    total_pnl = shares * S[:, -1] + cash - np.maximum(S[:, -1] - K, 0)
    total_var = np.var(total_pnl)

    if total_var > 0:
        disc_pct = float(np.var(disc_hedge) / total_var * 100)
        vol_pct = float(np.var(vol_misspec) / total_var * 100)
    else:
        disc_pct = vol_pct = 0.0

    return {
        'total_pnl_mean': float(np.mean(total_pnl)),
        'total_pnl_std': float(np.std(total_pnl)),
        'gamma_pnl_mean': float(np.mean(disc_hedge)),
        'theta_pnl_mean': float(np.mean(vol_misspec)),
        'discrete_rebalancing_pct': disc_pct,
        'vol_misspecification_pct': vol_pct,
        'gamma_variance_pct': disc_pct,
        'higher_order_pct': max(0.0, 100 - disc_pct - vol_pct),
    }


def vega_hedge_simulation(
    S0, K, T, r, q, sigma_initial, sigma_final,
    vol_shock_time=0.5, n_paths=10000, seed=42,
):
    """Simulate impact of a vol shock on hedged vs unhedged vega exposure."""
    if seed is not None:
        np.random.seed(seed)

    greeks_pre = black_scholes_greeks(S0, K, T * (1 - vol_shock_time), r, q, sigma_initial)
    vol_change_pct = (sigma_final - sigma_initial) * 100
    vega_pnl = greeks_pre['vega'] * vol_change_pct

    n_steps = 252
    dt = T / n_steps
    shock_step = int(vol_shock_time * n_steps)

    # Generate paths with vol regime change
    Z = np.random.standard_normal((n_paths, n_steps))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for i in range(n_steps):
        sig_t = sigma_initial if i < shock_step else sigma_final
        S[:, i + 1] = S[:, i] * np.exp(
            (r - q - 0.5 * sig_t**2) * dt + sig_t * np.sqrt(dt) * Z[:, i]
        )

    premium = float(black_scholes_greeks(S0, K, T, r, q, sigma_initial)['price'])
    cash = np.zeros(n_paths)
    shares = np.zeros(n_paths)

    for i in range(n_steps):
        tau = T - i * dt
        sig_t = sigma_initial if i < shock_step else sigma_final
        if tau > 1e-6:
            delta = black_scholes_greeks(S[:, i], K, tau, r, q, sig_t)['delta']
        else:
            delta = np.where(S[:, i] > K, 1.0, 0.0)
        trade = delta - shares
        cash = cash * np.exp(r * dt) - trade * S[:, i]
        shares = delta

    unhedged_pnl = premium + cash + shares * S[:, -1] - np.maximum(S[:, -1] - K, 0)

    return {
        'unhedged_pnl': unhedged_pnl,
        'vega_hedged_pnl': unhedged_pnl - vega_pnl,
        'vega_pnl_impact': float(vega_pnl),
        'vol_change': sigma_final - sigma_initial,
    }


if __name__ == "__main__":
    print("DELTA-GAMMA HEDGING COMPARISON")
    results = delta_gamma_hedge_simulation(
        S0=100, K1=100, K2=110, T=1.0, r=0.05, q=0.0,
        sigma_true=0.20, sigma_hedge=0.20, n_paths=50000,
        transaction_cost_bps=0.0,
    )
    print(f"Delta-only std:  {results['delta_only_pnl'].std():.4f}")
    print(f"Delta-gamma std: {results['delta_gamma_pnl'].std():.4f}")
    print(f"Std reduction:   {results['std_reduction_pct']:.1f}%")

    print()
    print("P&L ATTRIBUTION (correct vol)")
    attr = pnl_attribution_analysis(
        S0=100, K=100, T=0.25, r=0.05, q=0.0,
        sigma_true=0.20, sigma_hedge=0.20, n_paths=50000,
    )
    print(f"Discrete rebalancing: {attr['discrete_rebalancing_pct']:.1f}%")
    print(f"Vol misspecification: {attr['vol_misspecification_pct']:.1f}%")
    print(f"Higher-order terms:   {attr['higher_order_pct']:.1f}%")

    print()
    print("P&L ATTRIBUTION (vol misspec: 20% true, 25% hedge)")
    attr2 = pnl_attribution_analysis(
        S0=100, K=100, T=0.25, r=0.05, q=0.0,
        sigma_true=0.20, sigma_hedge=0.25, n_paths=50000,
    )
    print(f"Discrete rebalancing: {attr2['discrete_rebalancing_pct']:.1f}%")
    print(f"Vol misspecification: {attr2['vol_misspecification_pct']:.1f}%")
    print(f"Higher-order terms:   {attr2['higher_order_pct']:.1f}%")

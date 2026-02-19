"""Delta-hedging simulation engine."""

from typing import Dict, List, Optional

import numpy as np

from src import config
from src.pricing import BlackScholes


def run_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma_true: float,
    sigma_hedge: float,
    n_rebalances: int = 252,
    n_simulations: int = 10_000,
    option_type: str = "call",
    transaction_cost_bps: float = 0,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, float]:
    """Monte Carlo delta-hedging simulation. Returns dict of P&L stats."""
    np.random.seed(seed)

    dt = T / n_rebalances

    Z = np.random.standard_normal((n_simulations, n_rebalances))
    log_increments = (r - q - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_simulations, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    S_paths = S0 * np.exp(log_paths)

    bs0 = BlackScholes(S0, K, T, r, q, sigma_hedge)
    premium = float(bs0.price(option_type))

    delta_0 = float(bs0.delta(option_type))
    shares = np.full(n_simulations, delta_0)
    cash = np.full(n_simulations, premium - delta_0 * S0)

    max_inventory = np.abs(shares.copy())
    total_trades = np.zeros(n_simulations)
    total_tcosts = np.zeros(n_simulations)

    tc_frac = transaction_cost_bps / 10_000.0

    for i in range(1, n_rebalances):
        cash *= np.exp(r * dt)

        S_i = S_paths[:, i]
        tau = T - i * dt

        bs_i = BlackScholes(S_i, K, tau, r, q, sigma_hedge)
        new_delta = bs_i.delta(option_type).ravel()

        trade = new_delta - shares
        tcost = np.abs(trade) * S_i * tc_frac

        cash -= trade * S_i + tcost
        shares = new_delta

        max_inventory = np.maximum(max_inventory, np.abs(shares))
        total_trades += np.abs(trade)
        total_tcosts += tcost

    cash *= np.exp(r * dt)
    S_T = S_paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    # Liquidate shares + remaining cash - option payoff
    final_pnl = shares * S_T + cash - payoff

    mean_pnl = float(np.mean(final_pnl))
    std_pnl = float(np.std(final_pnl, ddof=1))
    median_pnl = float(np.median(final_pnl))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    pct_profitable = float(np.mean(final_pnl > 0) * 100.0)

    sorted_pnl = np.sort(final_pnl)
    cummax = np.maximum.accumulate(sorted_pnl[::-1])[::-1]
    max_drawdown = float(np.max(cummax - sorted_pnl))

    percentiles = {
        f"p{p}": float(np.percentile(final_pnl, p))
        for p in [5, 25, 50, 75, 95]
    }

    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "median_pnl": median_pnl,
        "sharpe": sharpe,
        "pct_profitable": pct_profitable,
        "max_drawdown": max_drawdown,
        "percentiles": percentiles,
        "final_pnl": final_pnl,
        "max_inventory": max_inventory,
        "total_trades": total_trades,
        "total_transaction_costs": total_tcosts,
    }


def single_path_detail(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma_true: float,
    sigma_hedge: float,
    n_rebalances: int = 252,
    option_type: str = "call",
    transaction_cost_bps: float = 0,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """Single-path delta hedge with full path-level detail for visualization."""
    np.random.seed(seed)

    dt = T / n_rebalances
    tc_frac = transaction_cost_bps / 10_000.0

    Z = np.random.standard_normal(n_rebalances)
    log_inc = (r - q - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * Z
    S_path = np.empty(n_rebalances + 1)
    S_path[0] = S0
    S_path[1:] = S0 * np.exp(np.cumsum(log_inc))

    delta_path = np.empty(n_rebalances + 1)
    shares_path = np.empty(n_rebalances + 1)
    cash_path = np.empty(n_rebalances + 1)
    portfolio_value = np.empty(n_rebalances + 1)
    option_value = np.empty(n_rebalances + 1)
    gamma_pnl_per_step = np.empty(n_rebalances)
    theta_pnl_per_step = np.empty(n_rebalances)

    bs0 = BlackScholes(S0, K, T, r, q, sigma_hedge)
    premium = float(bs0.price(option_type))
    delta_0 = float(bs0.delta(option_type))

    delta_path[0] = delta_0
    shares_path[0] = delta_0
    cash_path[0] = premium - delta_0 * S0
    portfolio_value[0] = shares_path[0] * S_path[0] + cash_path[0]
    option_value[0] = premium

    current_shares = delta_0
    current_cash = premium - delta_0 * S0

    for i in range(1, n_rebalances + 1):
        S_i = S_path[i]
        tau = T - i * dt

        # Greeks at previous node for P&L decomposition
        tau_prev = T - (i - 1) * dt
        bs_prev = BlackScholes(S_path[i - 1], K, tau_prev, r, q, sigma_hedge)
        gamma_prev = float(bs_prev.gamma())
        # theta is per calendar day from pricing module; scale to per-year then multiply by dt
        theta_prev = float(bs_prev.theta(option_type)) * 365.0

        dS = S_i - S_path[i - 1]
        gamma_pnl_per_step[i - 1] = 0.5 * gamma_prev * dS ** 2
        theta_pnl_per_step[i - 1] = theta_prev * dt

        current_cash *= np.exp(r * dt)

        if tau > 1e-12:
            bs_i = BlackScholes(S_i, K, tau, r, q, sigma_hedge)
            new_delta = float(bs_i.delta(option_type))
            opt_val = float(bs_i.price(option_type))
        else:
            new_delta = 0.0
            if option_type == "call":
                opt_val = max(S_i - K, 0.0)
            else:
                opt_val = max(K - S_i, 0.0)

        trade = new_delta - current_shares
        tcost = abs(trade) * S_i * tc_frac
        current_cash -= trade * S_i + tcost
        current_shares = new_delta

        delta_path[i] = new_delta
        shares_path[i] = current_shares
        cash_path[i] = current_cash
        portfolio_value[i] = current_shares * S_i + current_cash
        option_value[i] = opt_val

    hedge_pnl = portfolio_value - option_value

    return {
        "S_path": S_path,
        "delta_path": delta_path,
        "shares_path": shares_path,
        "cash_path": cash_path,
        "portfolio_value": portfolio_value,
        "option_value": option_value,
        "hedge_pnl": hedge_pnl,
        "gamma_pnl_per_step": gamma_pnl_per_step,
        "theta_pnl_per_step": theta_pnl_per_step,
    }


def gamma_pnl_decomposition(path_detail: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Decompose hedge P&L into gamma, theta, and residual components."""
    hedge_pnl = path_detail["hedge_pnl"]
    actual_pnl = np.diff(hedge_pnl)

    gamma_component = path_detail["gamma_pnl_per_step"]
    theta_component = path_detail["theta_pnl_per_step"]

    residual = actual_pnl - gamma_component - theta_component

    return {
        "actual_pnl": actual_pnl,
        "gamma_component": gamma_component,
        "theta_component": theta_component,
        "residual": residual,
    }

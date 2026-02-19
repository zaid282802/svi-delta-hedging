"""Numerical Greeks via central finite differences."""

from typing import Callable, Dict

import numpy as np
import pandas as pd

from src.pricing import BlackScholes


class NumericalGreeks:
    """Greeks via central finite differences, works with any pricing function."""

    @staticmethod
    def delta(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dS: float = 0.01,
    ) -> float:
        """Numerical delta."""
        price_up = pricer_func(S + dS, K, T, r, q, sigma, option_type)
        price_down = pricer_func(S - dS, K, T, r, q, sigma, option_type)
        return (price_up - price_down) / (2.0 * dS)

    @staticmethod
    def gamma(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dS: float = 0.01,
    ) -> float:
        """Numerical gamma."""
        price_up = pricer_func(S + dS, K, T, r, q, sigma, option_type)
        price_mid = pricer_func(S, K, T, r, q, sigma, option_type)
        price_down = pricer_func(S - dS, K, T, r, q, sigma, option_type)
        return (price_up - 2.0 * price_mid + price_down) / (dS * dS)

    @staticmethod
    def vega(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dSigma: float = 0.001,
    ) -> float:
        """Numerical vega per 1% vol move."""
        price_up = pricer_func(S, K, T, r, q, sigma + dSigma, option_type)
        price_down = pricer_func(S, K, T, r, q, sigma - dSigma, option_type)
        return (price_up - price_down) / (2.0 * dSigma) / 100.0

    @staticmethod
    def theta(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dT: float = 1.0 / 365.0,
    ) -> float:
        """Numerical theta (per calendar day)."""
        price_later = pricer_func(S, K, T - dT, r, q, sigma, option_type)
        price_now = pricer_func(S, K, T, r, q, sigma, option_type)
        return (price_later - price_now) / dT / 365.0

    @staticmethod
    def rho(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dR: float = 0.0001,
    ) -> float:
        """Numerical rho per 1% rate move."""
        price_up = pricer_func(S, K, T, r + dR, q, sigma, option_type)
        price_down = pricer_func(S, K, T, r - dR, q, sigma, option_type)
        return (price_up - price_down) / (2.0 * dR) / 100.0

    @staticmethod
    def vanna(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dS: float = 0.01,
        dSigma: float = 0.001,
    ) -> float:
        """Numerical vanna (cross-partial dS dsigma)."""
        f_pp = pricer_func(S + dS, K, T, r, q, sigma + dSigma, option_type)
        f_pm = pricer_func(S + dS, K, T, r, q, sigma - dSigma, option_type)
        f_mp = pricer_func(S - dS, K, T, r, q, sigma + dSigma, option_type)
        f_mm = pricer_func(S - dS, K, T, r, q, sigma - dSigma, option_type)
        return (f_pp - f_pm - f_mp + f_mm) / (4.0 * dS * dSigma)

    @staticmethod
    def volga(
        pricer_func: Callable,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        dSigma: float = 0.001,
    ) -> float:
        """Numerical volga (d2V/dsigma2)."""
        price_up = pricer_func(S, K, T, r, q, sigma + dSigma, option_type)
        price_mid = pricer_func(S, K, T, r, q, sigma, option_type)
        price_down = pricer_func(S, K, T, r, q, sigma - dSigma, option_type)
        return (price_up - 2.0 * price_mid + price_down) / (dSigma * dSigma)


def compare_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> pd.DataFrame:
    """Compare analytical vs numerical Greeks, returns DataFrame."""
    bs = BlackScholes(S, K, T, r, q, sigma)

    def bs_pricer(
        S_: float,
        K_: float,
        T_: float,
        r_: float,
        q_: float,
        sigma_: float,
        opt_type: str,
    ) -> float:
        return BlackScholes(S_, K_, T_, r_, q_, sigma_).price(opt_type)

    ng = NumericalGreeks

    greek_names = ["Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"]

    analytical_values = [
        bs.delta(option_type),
        bs.gamma(),
        bs.vega(),
        bs.theta(option_type),
        bs.rho(option_type),
        bs.vanna(),
        bs.volga(),
    ]

    numerical_values = [
        ng.delta(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.gamma(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.vega(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.theta(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.rho(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.vanna(bs_pricer, S, K, T, r, q, sigma, option_type),
        ng.volga(bs_pricer, S, K, T, r, q, sigma, option_type),
    ]

    abs_errors = [
        abs(a - n) for a, n in zip(analytical_values, numerical_values)
    ]
    rel_errors = [
        abs(a - n) / abs(a) * 100.0 if abs(a) > 1e-12 else 0.0
        for a, n in zip(analytical_values, numerical_values)
    ]

    return pd.DataFrame(
        {
            "Greek": greek_names,
            "Analytical": analytical_values,
            "Numerical": numerical_values,
            "Absolute Error": abs_errors,
            "Relative Error (%)": rel_errors,
        }
    )


def greek_surface_data(
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    S_range: np.ndarray,
    T_range: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Generate meshgrid data for 3D Greek surface plots."""
    S_grid, T_grid = np.meshgrid(S_range, T_range)
    n_rows, n_cols = S_grid.shape

    greek_names = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]
    surfaces: Dict[str, np.ndarray] = {
        name: np.zeros((n_rows, n_cols)) for name in greek_names
    }

    for i in range(n_rows):
        for j in range(n_cols):
            S_ij = float(S_grid[i, j])
            T_ij = float(T_grid[i, j])

            if T_ij < 1e-10:
                continue

            bs = BlackScholes(S_ij, K, T_ij, r, q, sigma)

            surfaces["delta"][i, j] = bs.delta(option_type)
            surfaces["gamma"][i, j] = bs.gamma()
            surfaces["vega"][i, j] = bs.vega()
            surfaces["theta"][i, j] = bs.theta(option_type)
            surfaces["rho"][i, j] = bs.rho(option_type)
            surfaces["vanna"][i, j] = bs.vanna()
            surfaces["volga"][i, j] = bs.volga()

    surfaces["S_grid"] = S_grid
    surfaces["T_grid"] = T_grid

    return surfaces

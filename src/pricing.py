"""Options pricing models: Black-Scholes, binomial trees, and Monte Carlo."""

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config


class BlackScholes:
    """Closed-form BS pricing with Greeks. Supports dividends and vectorized inputs."""

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0,
        sigma: float = 0.2,
    ) -> None:
        if np.any(np.asarray(S) <= 0):
            raise ValueError("Spot price S must be positive.")
        if np.any(np.asarray(K) <= 0):
            raise ValueError("Strike price K must be positive.")
        if np.any(np.asarray(T) < 0):
            raise ValueError("Time to expiry T must be non-negative.")
        if np.any(np.asarray(sigma) < 0):
            raise ValueError("Volatility sigma must be non-negative.")

        self.S = np.asarray(S)
        self.K = np.asarray(K)
        self.T = np.asarray(T)
        self.r = np.asarray(r)
        self.q = np.asarray(q)
        self.sigma = np.asarray(sigma)

    @property
    def d1(self):
        return (
            np.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)

    def _intrinsic(self, option_type: str = "call") -> np.ndarray:
        if option_type == "call":
            return np.maximum(self.S - self.K, 0.0)
        return np.maximum(self.K - self.S, 0.0)

    def _discounted_intrinsic(self, option_type: str = "call") -> np.ndarray:
        fwd = self.S * np.exp((self.r - self.q) * self.T)
        if option_type == "call":
            return np.exp(-self.r * self.T) * np.maximum(fwd - self.K, 0.0)
        return np.exp(-self.r * self.T) * np.maximum(self.K - fwd, 0.0)

    def _is_expired(self) -> bool:
        return bool(np.all(self.T == 0))

    def _is_zero_vol(self) -> bool:
        return bool(np.all(self.sigma < 1e-10))

    def call_price(self) -> np.ndarray:
        """Black-Scholes call price."""
        if self._is_expired():
            return self._intrinsic("call")
        if self._is_zero_vol():
            return self._discounted_intrinsic("call")

        d1, d2 = self.d1, self.d2
        return (
            self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
            - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

    def put_price(self) -> np.ndarray:
        """Black-Scholes put price."""
        if self._is_expired():
            return self._intrinsic("put")
        if self._is_zero_vol():
            return self._discounted_intrinsic("put")

        d1, d2 = self.d1, self.d2
        return (
            self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        )

    def price(self, option_type: str = "call") -> np.ndarray:
        """Call or put price."""
        if option_type == "call":
            return self.call_price()
        return self.put_price()

    def delta(self, option_type: str = "call") -> np.ndarray:
        """Call or put delta."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        discount = np.exp(-self.q * self.T)
        if option_type == "call":
            return discount * norm.cdf(self.d1)
        return discount * (norm.cdf(self.d1) - 1.0)

    def gamma(self) -> np.ndarray:
        """Gamma (same for calls and puts)."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        return (
            np.exp(-self.q * self.T)
            * norm.pdf(self.d1)
            / (self.S * self.sigma * np.sqrt(self.T))
        )

    def vega(self) -> np.ndarray:
        """Vega per 1% vol move."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        return (
            self.S
            * np.exp(-self.q * self.T)
            * norm.pdf(self.d1)
            * np.sqrt(self.T)
            / 100.0
        )

    def theta(self, option_type: str = "call") -> np.ndarray:
        """Theta per calendar day."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        d1, d2 = self.d1, self.d2
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        discount_r = np.exp(-self.r * self.T)

        term1 = -self.S * discount_q * norm.pdf(d1) * self.sigma / (2.0 * sqrt_T)

        if option_type == "call":
            result = (
                term1
                - self.r * self.K * discount_r * norm.cdf(d2)
                + self.q * self.S * discount_q * norm.cdf(d1)
            )
        else:
            result = (
                term1
                + self.r * self.K * discount_r * norm.cdf(-d2)
                - self.q * self.S * discount_q * norm.cdf(-d1)
            )

        return result / 365.0

    def rho(self, option_type: str = "call") -> np.ndarray:
        """Rho per 1% rate move."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        discount_r = np.exp(-self.r * self.T)
        if option_type == "call":
            return self.K * self.T * discount_r * norm.cdf(self.d2) / 100.0
        return -self.K * self.T * discount_r * norm.cdf(-self.d2) / 100.0

    def vanna(self) -> np.ndarray:
        """Vanna = d2V / (dS dsigma)."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        return -np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.d2 / self.sigma

    def volga(self) -> np.ndarray:
        """Volga (vomma) = d2V / dsigma2."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        vega_raw = (
            self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)
        )
        return vega_raw * self.d1 * self.d2 / self.sigma

    def charm(self, option_type: str = "call") -> np.ndarray:
        """Charm (delta decay)."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        d1, d2 = self.d1, self.d2
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)

        pdf_d1 = norm.pdf(d1)
        charm_common = discount_q * pdf_d1 * (
            2.0 * (self.r - self.q) * self.T - d2 * self.sigma * sqrt_T
        ) / (2.0 * self.T * self.sigma * sqrt_T)

        if option_type == "call":
            return -self.q * discount_q * norm.cdf(d1) + charm_common
        return self.q * discount_q * norm.cdf(-d1) + charm_common

    def speed(self) -> np.ndarray:
        """Speed (dGamma/dS)."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        g = self.gamma()
        sqrt_T = np.sqrt(self.T)
        return -(g / self.S) * (self.d1 / (self.sigma * sqrt_T) + 1.0)

    def color(self) -> np.ndarray:
        """Color (gamma decay)."""
        if self._is_expired() or self._is_zero_vol():
            return np.zeros_like(self.S)

        d1, d2 = self.d1, self.d2
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        pdf_d1 = norm.pdf(d1)

        return (
            -discount_q
            * pdf_d1
            / (2.0 * self.S * self.T * self.sigma * sqrt_T)
            * (
                2.0 * self.q * self.T
                + 1.0
                + d1
                * (2.0 * (self.r - self.q) * self.T - d2 * self.sigma * sqrt_T)
                / (self.sigma * sqrt_T)
            )
        )

    def all_greeks(self, option_type: str = "call") -> Dict[str, np.ndarray]:
        """Compute all Greeks at once and return as a dict."""
        return {
            "price": self.price(option_type),
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(option_type),
            "rho": self.rho(option_type),
            "vanna": self.vanna(),
            "volga": self.volga(),
            "charm": self.charm(option_type),
            "speed": self.speed(),
            "color": self.color(),
        }


def binomial_european(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int = config.DEFAULT_BINOMIAL_STEPS,
    option_type: str = "call",
) -> float:
    """Price a European option with the CRR binomial tree."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    asset_prices = S * u ** j * d ** (N - j)

    if option_type == "call":
        values = np.maximum(asset_prices - K, 0.0)
    else:
        values = np.maximum(K - asset_prices, 0.0)

    for _ in range(N):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

    return float(values[0])


def binomial_american(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int = config.DEFAULT_BINOMIAL_STEPS,
    option_type: str = "put",
) -> float:
    """Price an American option with the CRR binomial tree and early exercise."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    asset_prices = S * u ** j * d ** (N - j)

    if option_type == "call":
        values = np.maximum(asset_prices - K, 0.0)
    else:
        values = np.maximum(K - asset_prices, 0.0)

    for i in range(N - 1, -1, -1):
        j_i = np.arange(i + 1)
        asset_at_node = S * u ** j_i * d ** (i - j_i)

        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

        if option_type == "call":
            exercise = np.maximum(asset_at_node - K, 0.0)
        else:
            exercise = np.maximum(K - asset_at_node, 0.0)

        values = np.maximum(values, exercise)

    return float(values[0])


def monte_carlo_european(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = config.DEFAULT_N_MC_PATHS,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, float]:
    """Monte Carlo European pricing with antithetic + control variate variance reduction."""
    np.random.seed(seed)

    if antithetic:
        half = n_paths // 2
        z = np.random.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = np.random.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    S_T = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    disc = np.exp(-r * T)

    if control_variate:
        expected_S_T = S * np.exp((r - q) * T)
        cov_matrix = np.cov(payoffs, S_T)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        payoffs_adj = payoffs - beta * (S_T - expected_S_T)
    else:
        payoffs_adj = payoffs

    discounted = disc * payoffs_adj
    price = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))

    return {
        "price": price,
        "standard_error": se,
        "ci_lower": price - 1.96 * se,
        "ci_upper": price + 1.96 * se,
        "n_paths": len(discounted),
    }


def benchmark_methods(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> pd.DataFrame:
    """Compare BS, binomial, and MC prices side by side."""
    results = []

    t0 = time.perf_counter()
    bs = BlackScholes(S, K, T, r, q, sigma)
    bs_price = float(bs.price(option_type))
    t_bs = (time.perf_counter() - t0) * 1000.0
    results.append({
        "Method": "Black-Scholes",
        "Price": bs_price,
        "Error vs BS": 0.0,
        "95% CI": "N/A",
        "Time (ms)": round(t_bs, 3),
    })

    t0 = time.perf_counter()
    bin_price = binomial_european(S, K, T, r, q, sigma, option_type=option_type)
    t_bin = (time.perf_counter() - t0) * 1000.0
    results.append({
        "Method": f"Binomial (N={config.DEFAULT_BINOMIAL_STEPS})",
        "Price": bin_price,
        "Error vs BS": bin_price - bs_price,
        "95% CI": "N/A",
        "Time (ms)": round(t_bin, 3),
    })

    t0 = time.perf_counter()
    mc = monte_carlo_european(S, K, T, r, q, sigma, option_type=option_type)
    t_mc = (time.perf_counter() - t0) * 1000.0
    results.append({
        "Method": f"Monte Carlo ({config.DEFAULT_N_MC_PATHS:,} paths)",
        "Price": mc["price"],
        "Error vs BS": mc["price"] - bs_price,
        "95% CI": f"[{mc['ci_lower']:.4f}, {mc['ci_upper']:.4f}]",
        "Time (ms)": round(t_mc, 3),
    })

    return pd.DataFrame(results)

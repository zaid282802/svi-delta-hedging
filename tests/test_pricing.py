"""Pricing module tests."""

import pytest
import numpy as np
from src.pricing import BlackScholes, binomial_european, binomial_american, monte_carlo_european


S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20


class TestPutCallParity:
    """Verify C - P = S*exp(-qT) - K*exp(-rT) for European options."""

    def test_put_call_parity(self):
        bs = BlackScholes(S, K, T, r, q, sigma)
        call = float(bs.call_price())
        put = float(bs.put_price())
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(call - put - parity) < 1e-10

    def test_put_call_parity_otm(self):
        K_otm = 120.0
        bs = BlackScholes(S, K_otm, T, r, q, sigma)
        call = float(bs.call_price())
        put = float(bs.put_price())
        parity = S * np.exp(-q * T) - K_otm * np.exp(-r * T)
        assert abs(call - put - parity) < 1e-10


class TestBinomialConvergence:
    """Binomial tree with large N must converge to Black-Scholes."""

    def test_binomial_convergence_call(self):
        bs = BlackScholes(S, K, T, r, q, sigma)
        bs_call = float(bs.call_price())
        bin_call = binomial_european(S, K, T, r, q, sigma, N=1000, option_type="call")
        assert abs(bs_call - bin_call) < 0.01

    def test_binomial_convergence_put(self):
        bs = BlackScholes(S, K, T, r, q, sigma)
        bs_put = float(bs.put_price())
        bin_put = binomial_european(S, K, T, r, q, sigma, N=1000, option_type="put")
        assert abs(bs_put - bin_put) < 0.01


class TestMonteCarlo:
    """Monte Carlo pricing with variance reduction techniques."""

    def test_mc_within_confidence_interval(self):
        bs = BlackScholes(S, K, T, r, q, sigma)
        bs_call = float(bs.call_price())
        mc = monte_carlo_european(
            S, K, T, r, q, sigma,
            option_type="call",
            n_paths=500_000,
            antithetic=True,
            control_variate=True,
            seed=42,
        )
        # BS price must be within 2 standard errors of the MC estimate
        assert abs(mc["price"] - bs_call) < 2.0 * mc["standard_error"]

    def test_mc_antithetic_reduces_variance(self):
        mc_no_anti = monte_carlo_european(
            S, K, T, r, q, sigma,
            option_type="call",
            n_paths=100_000,
            antithetic=False,
            control_variate=False,
            seed=42,
        )
        mc_anti = monte_carlo_european(
            S, K, T, r, q, sigma,
            option_type="call",
            n_paths=100_000,
            antithetic=True,
            control_variate=False,
            seed=42,
        )
        assert mc_anti["standard_error"] < mc_no_anti["standard_error"]

    def test_mc_control_variate_reduces_variance(self):
        mc_no_cv = monte_carlo_european(
            S, K, T, r, q, sigma,
            option_type="call",
            n_paths=100_000,
            antithetic=False,
            control_variate=False,
            seed=42,
        )
        mc_cv = monte_carlo_european(
            S, K, T, r, q, sigma,
            option_type="call",
            n_paths=100_000,
            antithetic=False,
            control_variate=True,
            seed=42,
        )
        assert mc_cv["standard_error"] < mc_no_cv["standard_error"]


class TestDeepMoneyness:
    """Deep ITM/OTM boundary behavior."""

    def test_deep_itm_call(self):
        """Deep ITM call is approximately S*exp(-qT) - K*exp(-rT)."""
        S_deep = 200.0
        K_deep = 100.0
        bs = BlackScholes(S_deep, K_deep, T, r, q, sigma)
        call = float(bs.call_price())
        expected = S_deep * np.exp(-q * T) - K_deep * np.exp(-r * T)
        assert abs(call - expected) / expected < 0.01

    def test_deep_otm_put(self):
        """Deep OTM put (S >> K) is approximately zero."""
        S_deep = 200.0
        K_deep = 100.0
        bs = BlackScholes(S_deep, K_deep, T, r, q, sigma)
        put = float(bs.put_price())
        assert abs(put) < 0.01


class TestGreeksSanity:
    """Basic boundary and sign checks on Black-Scholes Greeks."""

    def test_atm_delta(self):
        """ATM call delta is close to 0.5 (above 0.5 due to positive drift r-q)."""
        bs = BlackScholes(100.0, 100.0, 0.5, r, q, sigma)
        delta = float(bs.delta("call"))
        assert abs(delta - 0.5) < 0.07

    def test_gamma_positive(self):
        """Gamma must be positive for all moneyness levels."""
        for S_val, K_val in [(80, 100), (100, 100), (120, 100)]:
            bs = BlackScholes(float(S_val), float(K_val), T, r, q, sigma)
            gamma = float(bs.gamma())
            assert gamma > 0, f"Gamma not positive for S={S_val}, K={K_val}"

    def test_call_delta_bounds(self):
        """Call delta is in [0, 1]."""
        for S_val in [80.0, 100.0, 120.0]:
            bs = BlackScholes(S_val, K, T, r, q, sigma)
            delta = float(bs.delta("call"))
            assert 0.0 <= delta <= 1.0, f"Call delta out of bounds for S={S_val}"

    def test_put_delta_bounds(self):
        """Put delta is in [-1, 0]."""
        for S_val in [80.0, 100.0, 120.0]:
            bs = BlackScholes(S_val, K, T, r, q, sigma)
            delta = float(bs.delta("put"))
            assert -1.0 <= delta <= 0.0, f"Put delta out of bounds for S={S_val}"

    def test_vega_positive(self):
        """Vega is positive for all options (call and put vega are identical)."""
        for S_val, K_val in [(80, 100), (100, 100), (120, 100)]:
            bs = BlackScholes(float(S_val), float(K_val), T, r, q, sigma)
            vega = float(bs.vega())
            assert vega > 0, f"Vega not positive for S={S_val}, K={K_val}"


class TestZeroTime:
    """At expiry (T=0), options equal intrinsic value."""

    def test_zero_time(self):
        for S_val, K_val in [(110.0, 100.0), (90.0, 100.0), (100.0, 100.0)]:
            bs = BlackScholes(S_val, K_val, 0.0, r, q, sigma)
            call = float(bs.call_price())
            put = float(bs.put_price())
            assert abs(call - max(S_val - K_val, 0.0)) < 1e-12
            assert abs(put - max(K_val - S_val, 0.0)) < 1e-12

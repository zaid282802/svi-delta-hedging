"""SVI calibration and vol surface tests."""

import pytest
import numpy as np
from src.vol_surface import (
    svi_total_variance,
    svi_implied_vol,
    calibrate_svi_slice,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
)


class TestSVICalibration:
    """Verify SVI calibration recovers known smiles."""

    def test_svi_recovers_flat_vol(self):
        """A flat smile (all IVs = 0.20) should be recovered with RMSE ~ 0."""
        flat_iv = 0.20
        T = 0.5
        forward = 100.0
        strikes = np.linspace(85, 115, 31)
        market_ivs = np.full_like(strikes, flat_iv)

        result = calibrate_svi_slice(strikes, market_ivs, forward, T)
        assert result["rmse_bps"] < 1.0, (
            f"RMSE too large for flat smile: {result['rmse_bps']:.2f} bps"
        )

    def test_svi_fit_quality(self):
        """Generate a synthetic smile from known SVI params, add 10 bps noise,
        recalibrate, and check RMSE < 20 bps."""
        T = 0.5
        forward = 100.0
        strikes = np.linspace(80, 120, 41)
        k = np.log(strikes / forward)

        a_true = 0.04
        b_true = 0.20
        rho_true = -0.30
        m_true = 0.0
        sigma_true = 0.10

        w_true = svi_total_variance(k, a_true, b_true, rho_true, m_true, sigma_true)
        iv_true = np.sqrt(np.maximum(w_true, 0.0) / T)

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 10.0 / 10_000, size=len(iv_true))
        iv_noisy = iv_true + noise

        result = calibrate_svi_slice(strikes, iv_noisy, forward, T)
        assert result["rmse_bps"] < 20.0, (
            f"RMSE too large after re-calibration: {result['rmse_bps']:.2f} bps"
        )


class TestButterflyArbitrage:
    """Butterfly no-arbitrage: g(k) >= 0 for typical equity SVI params."""

    def test_butterfly_no_arbitrage(self):
        """Typical equity SVI parameters (negative skew) should be arb-free."""
        a = 0.04
        b = 0.15
        rho = -0.50
        m = 0.0
        sigma = 0.15

        k_grid = np.linspace(-0.5, 0.5, 501)
        g, is_arb_free = check_butterfly_arbitrage(k_grid, a, b, rho, m, sigma)

        assert is_arb_free, (
            f"Butterfly arbitrage detected; min g(k) = {np.min(g):.6e}"
        )
        assert np.all(g >= -1e-10), "g(k) has values below tolerance"


class TestTotalVariancePositive:
    """Total variance w(k) must be positive for all k in [-1, 1]."""

    def test_total_variance_positive(self):
        a = 0.04
        b = 0.15
        rho = -0.50
        m = 0.0
        sigma = 0.15

        k_grid = np.linspace(-1.0, 1.0, 1001)
        w = svi_total_variance(k_grid, a, b, rho, m, sigma)
        assert np.all(w > 0), f"Non-positive total variance found; min w = {np.min(w):.6e}"


class TestCalendarArbitrage:
    """Calendar spread no-arbitrage: w(k, T1) <= w(k, T2) for T1 < T2."""

    def test_calendar_no_arbitrage(self):
        """Two SVI slices at T1 < T2, with T2 having larger 'a' level,
        must have w(T1) <= w(T2) everywhere."""
        params_T1 = {
            "a": 0.02,
            "b": 0.15,
            "rho": -0.40,
            "m": 0.0,
            "sigma": 0.15,
        }
        params_T2 = {
            "a": 0.06,
            "b": 0.15,
            "rho": -0.40,
            "m": 0.0,
            "sigma": 0.15,
        }

        T1, T2 = 0.25, 1.0
        surface_params = {T1: params_T1, T2: params_T2}

        result = check_calendar_arbitrage(surface_params)
        assert result["is_arb_free"], (
            f"Calendar arbitrage detected: {result['violations']}"
        )

        k_grid = np.linspace(-0.5, 0.5, 201)
        w1 = svi_total_variance(k_grid, **params_T1)
        w2 = svi_total_variance(k_grid, **params_T2)
        assert np.all(w1 <= w2 + 1e-10), "w(T1) > w(T2) for some k values"

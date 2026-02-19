"""Implied volatility solver tests."""

import pytest
import numpy as np
from src.pricing import BlackScholes
from src.implied_vol import implied_vol


T = 1.0
r = 0.05
q = 0.02


def _price_from_sigma(S, K, sigma_true, option_type="call"):
    bs = BlackScholes(S, K, T, r, q, sigma_true)
    return float(bs.price(option_type))


class TestIVRecovery:
    """Price with known sigma, then recover it via implied_vol()."""

    def test_iv_recovery_atm(self):
        S_val, K_val, sigma_true = 100.0, 100.0, 0.25
        price = _price_from_sigma(S_val, K_val, sigma_true)
        iv = implied_vol(price, S_val, K_val, T, r, q, "call")
        assert abs(iv - sigma_true) < 1e-6

    def test_iv_recovery_itm(self):
        S_val, K_val, sigma_true = 110.0, 100.0, 0.30
        price = _price_from_sigma(S_val, K_val, sigma_true)
        iv = implied_vol(price, S_val, K_val, T, r, q, "call")
        assert abs(iv - sigma_true) < 1e-6

    def test_iv_recovery_otm(self):
        S_val, K_val, sigma_true = 90.0, 100.0, 0.30
        price = _price_from_sigma(S_val, K_val, sigma_true)
        iv = implied_vol(price, S_val, K_val, T, r, q, "call")
        assert abs(iv - sigma_true) < 1e-6


class TestIVExtremes:
    """Solver should handle very low and very high volatility inputs."""

    def test_iv_low_vol(self):
        S_val, K_val, sigma_true = 100.0, 100.0, 0.05
        price = _price_from_sigma(S_val, K_val, sigma_true)
        iv = implied_vol(price, S_val, K_val, T, r, q, "call")
        assert abs(iv - sigma_true) < 1e-6

    def test_iv_high_vol(self):
        S_val, K_val, sigma_true = 100.0, 100.0, 1.50
        price = _price_from_sigma(S_val, K_val, sigma_true)
        iv = implied_vol(price, S_val, K_val, T, r, q, "call")
        assert abs(iv - sigma_true) < 1e-6


class TestIVInvalid:
    """Solver returns NaN for prices below intrinsic."""

    def test_iv_nan_for_bad_price(self):
        S_val, K_val = 110.0, 100.0
        intrinsic = max(S_val * np.exp(-q * T) - K_val * np.exp(-r * T), 0.0)
        bad_price = intrinsic * 0.5
        iv = implied_vol(bad_price, S_val, K_val, T, r, q, "call")
        assert np.isnan(iv), "Expected NaN for price below intrinsic"

"""Greeks tests: analytical vs numerical comparison."""

import pytest
import numpy as np
from src.pricing import BlackScholes
from src.greeks import NumericalGreeks


def bs_pricer(S, K, T, r, q, sigma, option_type):
    return BlackScholes(S, K, T, r, q, sigma).price(option_type)


T = 0.5
r = 0.05
q = 0.02
sigma = 0.25

MONEYNESS = [
    ("ITM", 110.0, 100.0),
    ("ATM", 100.0, 100.0),
    ("OTM", 90.0, 100.0),
]

OPTION_TYPES = ["call", "put"]

_PARAMS = [
    (label, S_val, K_val, opt)
    for label, S_val, K_val in MONEYNESS
    for opt in OPTION_TYPES
]

_IDS = [f"{label}-{opt}" for label, _, _, opt in _PARAMS]

REL_TOL = 0.001  # 0.1 %


def _rel_error(analytical, numerical):
    if abs(analytical) < 1e-12:
        return abs(numerical)
    return abs(analytical - numerical) / abs(analytical)


class TestDelta:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_delta(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.delta(opt))
        numerical = NumericalGreeks.delta(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Delta mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestGamma:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_gamma(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.gamma())
        numerical = NumericalGreeks.gamma(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Gamma mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestVega:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_vega(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.vega())
        numerical = NumericalGreeks.vega(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Vega mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestTheta:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_theta(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.theta(opt))
        numerical = NumericalGreeks.theta(
            bs_pricer, S_val, K_val, T, r, q, sigma, opt, dT=1e-4,
        )
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Theta mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestRho:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_rho(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.rho(opt))
        numerical = NumericalGreeks.rho(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Rho mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestVanna:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_vanna(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.vanna())
        numerical = NumericalGreeks.vanna(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Vanna mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )


class TestVolga:

    @pytest.mark.parametrize("label,S_val,K_val,opt", _PARAMS, ids=_IDS)
    def test_volga(self, label, S_val, K_val, opt):
        bs = BlackScholes(S_val, K_val, T, r, q, sigma)
        analytical = float(bs.volga())
        numerical = NumericalGreeks.volga(bs_pricer, S_val, K_val, T, r, q, sigma, opt)
        assert _rel_error(analytical, numerical) < REL_TOL, (
            f"Volga mismatch ({label} {opt}): analytical={analytical}, numerical={numerical}"
        )

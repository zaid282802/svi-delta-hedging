"""Tests for enhancement modules."""

import pytest
import numpy as np


class TestEnhancedHedging:

    def test_black_scholes_greeks_atm(self):
        from src.enhanced_hedging import black_scholes_greeks

        greeks = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=0.20)

        assert 0.4 < greeks['delta'] < 0.6
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
        assert greeks['theta'] < 0

    def test_black_scholes_greeks_edge_cases(self):
        from src.enhanced_hedging import black_scholes_greeks

        greeks = black_scholes_greeks(S=105, K=100, T=0, r=0.05, q=0.02, sigma=0.20)
        assert greeks['price'] == 5.0
        assert greeks['delta'] == 1.0

        with pytest.raises(ValueError):
            black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, q=0.02, sigma=-0.20)

    def test_delta_gamma_hedge_reduces_variance(self):
        from src.enhanced_hedging import delta_gamma_hedge_simulation

        results = delta_gamma_hedge_simulation(
            S0=100, K1=100, K2=105, T=0.25, r=0.05, q=0.02,
            sigma_true=0.20, sigma_hedge=0.20, n_paths=1000, seed=42,
        )

        assert results['std_reduction_pct'] > 0
        assert len(results['delta_only_pnl']) == 1000
        assert len(results['delta_gamma_pnl']) == 1000

    def test_pnl_attribution_sums_correctly(self):
        from src.enhanced_hedging import pnl_attribution_analysis

        attr = pnl_attribution_analysis(
            S0=100, K=100, T=0.25, r=0.05, q=0.02,
            sigma_true=0.20, sigma_hedge=0.20, n_paths=1000, seed=42,
        )

        assert attr['discrete_rebalancing_pct'] >= 0
        assert attr['vol_misspecification_pct'] >= 0
        assert attr['higher_order_pct'] >= 0


class TestHistoricalCaseStudies:

    def test_covid_crash_case_study(self):
        from src.historical_case_studies import covid_crash_march_2020

        case = covid_crash_march_2020()

        assert case.event_name == "COVID-19 Market Crash"
        assert case.iv_after > case.iv_before
        assert case.delta_hedge_pnl < 0
        assert len(case.key_lessons) >= 3

    def test_earnings_case_study(self):
        from src.historical_case_studies import earnings_iv_crush_example

        case = earnings_iv_crush_example()

        assert case.iv_after < case.iv_before
        assert case.vega_impact > 0

    def test_summary_table_generation(self):
        from src.historical_case_studies import generate_case_study_summary

        df = generate_case_study_summary()

        assert len(df) == 3
        assert 'Event' in df.columns
        assert 'IV Before' in df.columns


class TestModelRisk:

    def test_model_risk_report_generation(self):
        from src.model_risk_analysis import generate_model_risk_report

        risks = generate_model_risk_report()

        assert len(risks) >= 5
        assert all(r.typical_impact_bps > 0 for r in risks)
        assert all(r.stress_impact_bps >= r.typical_impact_bps for r in risks)

    def test_monte_carlo_risk_analysis(self):
        from src.model_risk_analysis import analyze_monte_carlo_risk

        result = analyze_monte_carlo_risk(
            price_estimate=9.26, standard_error=0.004, n_paths=500000,
        )

        assert result['ci_lower'] < result['price'] < result['ci_upper']
        assert result['relative_se_pct'] < 1


class TestPerformanceBenchmarks:

    def test_bs_call_correctness(self):
        from src.performance_benchmarks import bs_call

        price = bs_call(100, 100, 0.25, 0.05, 0.02, 0.20)
        assert 3.5 < price < 5.5

    def test_binomial_convergence(self):
        from src.performance_benchmarks import bs_call, binomial_call

        bs_price = bs_call(100, 100, 0.25, 0.05, 0.02, 0.20)
        bin_price = binomial_call(100, 100, 0.25, 0.05, 0.02, 0.20, N=1000)

        assert abs(bs_price - bin_price) < 0.01

    def test_benchmark_results_structure(self):
        from src.performance_benchmarks import run_benchmarks, results_to_dataframe

        results = run_benchmarks()
        df = results_to_dataframe(results)

        assert len(results) >= 5
        assert 'Operation' in df.columns
        assert 'Time (ms)' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

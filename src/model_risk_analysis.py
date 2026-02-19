"""Model risk analysis: quantifies risk from BSM assumptions, SVI calibration, MC estimation."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from scipy.stats import norm


@dataclass
class ModelRiskItem:
    assumption: str
    risk_type: str
    typical_impact_bps: float
    stress_impact_bps: float
    mitigation: str
    example: str


def generate_model_risk_report():
    """Comprehensive model risk report across all framework assumptions."""
    return [
        ModelRiskItem(
            "Constant Volatility (BSM)", "Volatility Risk", 50, 500,
            "Use stochastic vol models (Heston, SABR); recalibrate frequently",
            "March 2020: Realized vol spiked from 15% to 80%, causing ~$25 hedge loss per ATM option",
        ),
        ModelRiskItem(
            "Log-normal Returns (BSM)", "Distribution Risk", 20, 300,
            "Use fat-tailed distributions; incorporate jump diffusion",
            "Flash crash 2010: 9% move in 5 minutes impossible under log-normal",
        ),
        ModelRiskItem(
            "Continuous Trading", "Discrete Hedging Risk", 10, 200,
            "Hedge more frequently; use gamma-reducing overlays",
            "Daily vs hourly hedging: P&L std increases ~40% for ATM options",
        ),
        ModelRiskItem(
            "No Transaction Costs", "Execution Risk", 5, 50,
            "Incorporate costs into hedge ratios; use Whalley-Wilmott bandwidth",
            "5 bps round-trip on 100 rebalances = 50 bps cumulative drag",
        ),
        ModelRiskItem(
            "SVI Parameterization", "Calibration Risk", 30, 100,
            "Use arbitrage-free SVI (SSVI); validate butterfly/calendar constraints",
            "Extreme wings: SVI can produce negative variance if parameters extrapolate poorly",
        ),
        ModelRiskItem(
            "Static IV Surface", "Smile Dynamics Risk", 25, 150,
            "Model smile dynamics (sticky strike vs sticky delta); recalibrate intraday",
            "2% spot move: Smile shift causes ~15 bps mispricing if not recalibrated",
        ),
        ModelRiskItem(
            "Monte Carlo Sampling", "Estimation Risk", 10, 30,
            "Use variance reduction (antithetic, control variate); increase paths",
            "500K paths with CV: SE = $0.004 (0.04% of $10 option)",
        ),
    ]


def analyze_monte_carlo_risk(price_estimate, standard_error, n_paths, confidence_level=0.95):
    """Quantify MC estimation risk: CI width, relative SE, paths needed for target precision."""
    z = norm.ppf((1 + confidence_level) / 2)
    ci_half = z * standard_error
    return {
        'price': price_estimate,
        'standard_error': standard_error,
        'relative_se_pct': (standard_error / price_estimate) * 100 if price_estimate > 0 else 0,
        'ci_lower': price_estimate - ci_half,
        'ci_upper': price_estimate + ci_half,
        'ci_width_pct': (2 * ci_half / price_estimate) * 100 if price_estimate > 0 else 0,
        'paths_for_0.1pct_se': int(
            (standard_error / (price_estimate * 0.001))**2 * n_paths
        ) if price_estimate > 0 else n_paths,
    }


def model_risk_to_dataframe():
    """Convert model risk report to DataFrame for display."""
    risks = generate_model_risk_report()
    return pd.DataFrame([
        {
            'Assumption': r.assumption,
            'Risk Type': r.risk_type,
            'Typical (bps)': r.typical_impact_bps,
            'Stress (bps)': r.stress_impact_bps,
            'Mitigation': r.mitigation,
        }
        for r in risks
    ])


def print_model_risk_report():
    risks = generate_model_risk_report()

    print("MODEL RISK ANALYSIS REPORT")

    for i, r in enumerate(risks, 1):
        print(f"\n{i}. {r.assumption}")
        print(f"   Risk Type: {r.risk_type}")
        print(f"   Impact: {r.typical_impact_bps:.0f} bps typical | {r.stress_impact_bps:.0f} bps stress")
        print(f"   Mitigation: {r.mitigation}")
        print(f"   Example: {r.example}")

    total_typical = sum(r.typical_impact_bps for r in risks)
    total_stress = sum(r.stress_impact_bps for r in risks)

    print()
    print(f"TOTAL MODEL RISK: {total_typical:.0f} bps typical | {total_stress:.0f} bps stress")
    print("Note: Risks not simply additive due to correlations.")


if __name__ == "__main__":
    print_model_risk_report()
    print("\n")
    print(model_risk_to_dataframe().to_string(index=False))

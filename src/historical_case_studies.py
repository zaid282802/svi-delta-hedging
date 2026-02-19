"""Historical case studies: COVID crash, earnings IV crush, flash crash."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class CaseStudyResult:
    """Container for case study analysis results."""
    event_name: str
    date: str
    spot_before: float
    spot_after: float
    strike: float
    days_to_expiry: int
    iv_before: float
    iv_after: float
    realized_vol: float
    delta_hedge_pnl: float
    vega_impact: float
    gamma_impact: float
    key_lessons: List[str]


def covid_crash_march_2020():
    """March 2020 COVID crash. SPY fell 34% in 23 trading days, VIX hit 82.7."""
    spot_before, spot_after, strike = 339.08, 218.26, 340.0
    iv_before, iv_after = 0.13, 0.82
    realized_vol = 1.16

    avg_gamma = 0.015
    spot_move = spot_before - spot_after
    gamma_pnl = -0.5 * avg_gamma * spot_move**2

    vega_per_point = 0.35
    vega_pnl = -vega_per_point * (iv_after - iv_before) * 100

    return CaseStudyResult(
        event_name="COVID-19 Market Crash",
        date="Feb 19 - Mar 23, 2020",
        spot_before=spot_before, spot_after=spot_after,
        strike=strike, days_to_expiry=30,
        iv_before=iv_before, iv_after=iv_after,
        realized_vol=realized_vol,
        delta_hedge_pnl=gamma_pnl + vega_pnl,
        vega_impact=vega_pnl, gamma_impact=gamma_pnl,
        key_lessons=[
            "Vol misspecification was extreme: hedging at 13% IV while realized was 116%",
            "Discrete hedging fails during gap moves (market fell 10%+ on multiple days)",
            "Vega exposure was the dominant P&L driver, not gamma",
            "Risk limits and position sizing are critical for tail events",
        ],
    )


def earnings_iv_crush_example():
    """NVDA-style earnings IV crush. IV drops 30-50% post-announcement."""
    spot_before, spot_after, strike = 500.0, 520.0, 500.0
    iv_before, iv_after = 0.65, 0.38

    implied_move = 0.065
    straddle_premium = spot_before * implied_move
    straddle_intrinsic = max(spot_after - strike, 0)
    straddle_pnl = straddle_premium - straddle_intrinsic

    vega_per_point = 0.15
    vega_gain = vega_per_point * (iv_before - iv_after) * 100

    return CaseStudyResult(
        event_name="Earnings IV Crush (NVDA-style)",
        date="Stylized Example",
        spot_before=spot_before, spot_after=spot_after,
        strike=strike, days_to_expiry=7,
        iv_before=iv_before, iv_after=iv_after,
        realized_vol=0.04 * np.sqrt(365 / 7),
        delta_hedge_pnl=straddle_pnl * 0.7,
        vega_impact=vega_gain, gamma_impact=-5.0,
        key_lessons=[
            "IV typically overestimates actual earnings moves",
            "Straddle selling around earnings is popular but has negative skew risk",
            "The 27-point IV crush provided profit on top of theta decay",
            "Tail events (earnings disasters) can lose 3-5x the premium collected",
        ],
    )


def flash_crash_may_2010():
    """May 6, 2010 Flash Crash. Dow dropped 998 points in 5 minutes then recovered."""
    spot_before, spot_low, spot_close = 117.0, 106.0, 114.0
    strike = 117.0
    iv_before, iv_peak, iv_close = 0.22, 0.45, 0.28

    gamma = 0.02
    max_move = spot_before - spot_low
    gamma_pnl = -0.5 * gamma * max_move**2

    return CaseStudyResult(
        event_name="Flash Crash",
        date="May 6, 2010",
        spot_before=spot_before, spot_after=spot_close,
        strike=strike, days_to_expiry=14,
        iv_before=iv_before, iv_after=iv_close,
        realized_vol=iv_peak * 2,
        delta_hedge_pnl=gamma_pnl * 2,
        vega_impact=-0.50, gamma_impact=gamma_pnl,
        key_lessons=[
            "Market fell 9.2% in 5 minutes - too fast to rebalance",
            "Bid-ask spreads blew out; hedging was literally impossible",
            "Some options traded at $0.01 (later canceled)",
            "Continuous hedging is a fiction; markets gap",
        ],
    )


def generate_case_study_summary():
    """Summary DataFrame of all case studies."""
    cases = [covid_crash_march_2020(), earnings_iv_crush_example(), flash_crash_may_2010()]
    rows = []
    for c in cases:
        rows.append({
            'Event': c.event_name,
            'Date': c.date,
            'Spot Move': f"{c.spot_before:.0f} -> {c.spot_after:.0f}",
            'IV Before': f"{c.iv_before * 100:.0f}%",
            'IV After': f"{c.iv_after * 100:.0f}%",
            'Hedge P&L': f"${c.delta_hedge_pnl:.2f}",
            'Vega Impact': f"${c.vega_impact:.2f}",
            'Gamma Impact': f"${c.gamma_impact:.2f}",
        })
    return pd.DataFrame(rows)


def print_case_study(case):
    print()
    print(f"CASE STUDY: {case.event_name}")
    print(f"Date: {case.date}")
    print(f"Spot: ${case.spot_before:.2f} -> ${case.spot_after:.2f}")
    print(f"IV: {case.iv_before * 100:.1f}% -> {case.iv_after * 100:.1f}%")
    print(f"Realized Vol (annualized): {case.realized_vol * 100:.1f}%")
    print(f"\nP&L Breakdown:")
    print(f"  Delta Hedge P&L: ${case.delta_hedge_pnl:.2f}")
    print(f"  Vega Impact:     ${case.vega_impact:.2f}")
    print(f"  Gamma Impact:    ${case.gamma_impact:.2f}")
    print(f"\nKey Lessons:")
    for i, lesson in enumerate(case.key_lessons, 1):
        print(f"  {i}. {lesson}")


if __name__ == "__main__":
    print_case_study(covid_crash_march_2020())
    print_case_study(earnings_iv_crush_example())
    print_case_study(flash_crash_may_2010())

    print()
    print("SUMMARY TABLE")
    print(generate_case_study_summary().to_string(index=False))

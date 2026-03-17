# Regime-Dependent Delta Hedging with SVI-Calibrated Volatility Surfaces on SPX Index Options

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-80%20passing-green.svg)]()

**Author:** [Zaid Annigeri](https://linkedin.com/in/zaidannigeri) | Master of Quantitative Finance, Rutgers Business School

> SVI is the industry standard for volatility surface calibration. The literature validates
> fitting quality (10--50 bps RMSE) but stops there. **No study has tested whether a better
> surface fit produces a better hedge.** This thesis closes that loop on 2,000 real SPX options
> across four VIX regimes.

**Presented at:** Future Alpha 2026 | Brooklyn Marriott | March 31 -- April 1, 2026

## Key Results

| Metric | Value |
|---|---|
| Sample | 28.6M SPX options → 5.6M filtered → 2,000 stratified (500/regime) |
| Period | January 2019 -- December 2024 |
| SVI RMSE | 19.5 bps median |
| Arb-free rate | 68.6% |
| Best overall | Close-to-Close RV (+5.8%, p = 0.008) |
| Worst overall | Parkinson RV (−43.2%) |
| Delta-Gamma hedge | 46% std reduction |

### Three Hypotheses

- **H1:** SVI reduces hedging error vs. flat BSM → **REJECTED** (−9.4%, p < 0.001)
- **H2:** Yang-Zhang beats Close-to-Close RV → **REJECTED**
- **H3:** Optimal vol input is regime-dependent → **PARTIAL SUPPORT**

### The Counterintuitive Finding

The industry-standard volatility surface makes hedging *worse*. Calibration noise from the 5-parameter SVI optimization, interpolation staleness (calibrating every 5th day), and low signal-to-noise for most strikes overwhelm the smile information. Only OTM calls have steep enough slope to benefit from SVI.

### Practitioner Takeaway

| Segment | Best Vol Input | Gain |
|---|---|---|
| OTM Calls | SVI Surface IV | +6--12% |
| ATM | Flat BSM IV | baseline |
| OTM Puts | CC Realized Vol | +21--48% |
| VIX ≥ 35 | Focus on gamma management | — |

No single volatility input wins everywhere. A moneyness-conditional strategy outperforms any single approach.

## Project Structure

```
options-vol-surface/
├── thesis_project/
│   ├── src/                           # Pipeline scripts (run in order)
│   │   ├── run_hedging_backtest.py    # 1. Hedging backtest across vol inputs
│   │   ├── run_svi_calibration.py     # 2. SVI surface calibration
│   │   ├── run_butterfly_check.py     # 3. Butterfly arbitrage validation
│   │   └── run_analysis.py            # 4. Statistical analysis & tables
│   │
│   ├── report/                        # Academic deliverables
│   │   ├── thesis.tex                 # 40-page thesis (LaTeX source)
│   │   ├── thesis.pdf                 # Compiled thesis
│   │   ├── defense_qa.tex             # Defense Q&A preparation
│   │   └── defense_qa.pdf
│   │
│   ├── results/
│   │   ├── figures/                   # 8 thesis figures (PNG + PDF, 600 DPI)
│   │   ├── tables/                    # 6 result CSVs
│   │   ├── handout/                   # Conference handout (1-page landscape)
│   │   │   ├── handout_v2.tex
│   │   │   ├── gen_charts.py          # Generates the 4 handout charts
│   │   │   ├── qr_github.png
│   │   │   └── figures/               # 4 handout chart PNGs
│   │   ├── poster/                    # Conference poster + designer assets
│   │   │   ├── poster_content.tex
│   │   │   ├── poster_content.pdf
│   │   │   ├── 9 × fig_poster_*.png
│   │   │   ├── share/                 # Designer handoff package v1
│   │   │   └── ShareV1.2/             # Designer handoff package v1.2
│   │   ├── slides/                    # Presentation slides + 4 figures
│   │   ├── one_pager/                 # Extended abstract
│   │   └── pitch/                     # Pitch scripts
│   │
│   ├── data/                          # NOT in repo (see Data section below)
│   │   ├── raw/                       # WRDS CSVs + VIX history
│   │   └── processed/                 # Parquet output files
│   │
│   ├── existing_code/                 # Original options-vol-surface project
│   │   ├── src/                       # Pricing engine, Greeks, SVI, hedging (12 modules)
│   │   ├── tests/                     # 80 unit tests
│   │   ├── notebooks/                 # 4 Jupyter notebooks
│   │   └── report/                    # Original project report
│   │
│   └── .project-notes/               # Internal project documentation
│       ├── project-notes.md           # Key numbers, rules, glossary
│       ├── context/                   # Methodology, results summary, specs
│       └── scripts/                   # Figure generation scripts (matplotlib)
│
└── requirements.txt
```

## Methodology

### Data Pipeline

1. **Source:** 28.6M SPX options from WRDS OptionMetrics (Jan 2019 -- Dec 2024)
2. **Filtering:** Volume ≥ 10, 7 ≤ DTE ≤ 180, moneyness 0.8--1.2, midpoint > $0.50 → 5.6M
3. **Stratification:** 2,000 options sampled (500 per VIX regime: Low < 15, Normal 15--25, High 25--35, Crisis ≥ 35)

### Volatility Inputs Tested

| Input | Description |
|---|---|
| Flat BSM IV | OptionMetrics implied volatility (benchmark) |
| SVI Surface IV | Gatheral's 5-parameter SVI, calibrated per expiration slice |
| CC Realized Vol | 21-day close-to-close, annualized √252 |
| Parkinson RV | High-low range estimator (5× statistical efficiency under GBM) |
| Yang-Zhang RV | OHLC estimator combining overnight jumps with intraday range |

### Hedging Protocol

- BSM delta computed from each vol input
- Daily rebalancing to expiry
- Performance measured by Hull & White (2017) Gain statistic: `Gain = 1 − (std_method / std_benchmark)`
- Statistical significance via Levene's test for variance equality

### P&L Attribution

El Karoui, Jeanblanc-Picqué & Shreve (1998) decomposition:
- **Discrete rebalancing error** (−38% to −59% of variance)
- **Volatility misspecification** (−20% to +6%)
- **Higher-order residual** (153% to 162% — dominates across all regimes)

## Data

Raw data is **not included** in this repository due to size (3.5 GB) and WRDS licensing restrictions.

To reproduce:
1. Obtain access to [WRDS OptionMetrics](https://wrds-www.wharton.upenn.edu/)
2. Download SPX options data for 2019--2024
3. Place CSVs in `thesis_project/data/raw/`
4. Run the pipeline (see below)

## Installation

```bash
git clone https://github.com/zaid282802/options-vol-surface.git
cd options-vol-surface
pip install -r requirements.txt
```

## Usage

### Run the thesis pipeline

```bash
cd thesis_project

# Run in order — each step depends on the previous
python src/run_hedging_backtest.py
python src/run_svi_calibration.py
python src/run_butterfly_check.py
python src/run_analysis.py
```

### Run the original project tests

```bash
cd thesis_project/existing_code
pytest tests/ -v
```

### Regenerate figures

```bash
# Thesis figures (8 figures, PNG + PDF)
python .project-notes/scripts/generate_figs_1_to_4.py
python .project-notes/scripts/generate_figures_5_to_8.py

# Poster figures (9 figures)
python .project-notes/scripts/generate_poster_figures.py

# Slide figures (4 figures)
python .project-notes/scripts/generate_slide_figures.py

# Handout charts (4 figures)
cd results/handout && python gen_charts.py
```

## Core Library (existing_code/src/)

The thesis builds on a from-scratch options pricing library:

- **pricing.py** — Black-Scholes analytical, CRR binomial tree, Monte Carlo with antithetic + control variate variance reduction
- **greeks.py** — 10 analytical Greeks + numerical finite differences (< 0.1% relative error)
- **implied_vol.py** — Newton-Raphson IV solver with Brenner-Subrahmanyam seed, bisection fallback
- **vol_surface.py** — SVI raw parameterization, DE + L-BFGS-B two-stage calibration, butterfly & calendar arb checks
- **realized_vol.py** — Close-to-close, Parkinson, Yang-Zhang estimators
- **delta_hedge.py** — Monte Carlo delta-hedge simulation with gamma-theta decomposition
- **enhanced_hedging.py** — Delta-gamma hedging, P&L variance attribution

## References

- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI Volatility Surfaces. *Quantitative Finance*, 14(1), 59-71.
- Hull, J. & White, A. (2017). Optimal Delta Hedging for Options. *Journal of Banking & Finance*, 82, 180-190.
- El Karoui, N., Jeanblanc-Picqué, M. & Shreve, S. (1998). Robustness of the Black and Scholes Formula. *Mathematical Finance*, 8(2), 93-126.
- Ruf, J. & Wang, W. (2022). Hedging with Linear Regressions and Neural Networks. *Journal of Business & Economic Statistics*, 40(4).
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. *Journal of Business*, 53(1), 61-65.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.

## Contact

**Zaid Annigeri** — ma.zaid@rutgers.edu | [LinkedIn](https://linkedin.com/in/zaidannigeri) | [GitHub](https://github.com/zaid282802)

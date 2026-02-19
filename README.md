# Options Pricing, Greeks & Volatility Surface

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-80%20passing-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

**Author:** [Zaid Annigeri](https://linkedin.com/in/zed228) | Master of Quantitative Finance, Rutgers Business School

> European options pricing engine with SVI implied volatility surface calibration,
> delta-gamma hedging simulation, and P&L attribution analysis on SPY options,
> demonstrating the gamma-theta tradeoff that drives options market making.

## Key Results

### Pricing Engine Accuracy

Benchmark parameters: S = K = 100, T = 0.25y, r = 5%, q = 0%, sigma = 20%.

| Method | Call Price | Put Price | Error vs BS | Compute Time |
|---|---:|---:|---:|---:|
| Black-Scholes (analytical) | $4.6150 | $3.3728 | -- | <1 ms |
| Binomial CRR (N=1,000) | $4.6140 | $3.3718 | -$0.0010 | ~19 ms |
| Monte Carlo (500K paths) | $4.6157 | $3.3735 | +$0.0007 | ~67 ms |

Monte Carlo uses antithetic variates and control variate (forward price) variance reduction,
achieving standard error < $0.005 on 500,000 paths.

### Greeks Validation

All Greeks validated to < 0.1% relative error via bump-and-revalue central finite differences.

| Greek | Analytical | Numerical | Rel. Error (%) | Description |
|---|---:|---:|---:|---|
| Delta | 0.5695 | 0.5695 | < 0.0001 | dV/dS |
| Gamma | 0.0393 | 0.0393 | < 0.0001 | d2V/dS2 |
| Vega | 0.1964 | 0.1964 | 0.0001 | dV/dsigma (per 1 vol-pt) |
| Theta | -0.0287 | -0.0288 | 0.2108 | dV/dT (per calendar day) |
| Rho | 0.1308 | 0.1308 | < 0.0001 | dV/dr (per 1 rate-pt) |
| Vanna | -0.1473 | -0.1473 | 0.0040 | d2V/dS dsigma |
| Volga | 1.2891 | 1.2892 | 0.0029 | d2V/dsigma2 |

### Implied Volatility Surface

SVI (Stochastic Volatility Inspired) parameterization calibrated per expiration slice using
differential evolution global search followed by L-BFGS-B local refinement. Typical fit quality:
RMSE 18-24 bps, R-squared > 0.999. All butterfly and calendar arbitrage conditions satisfied.

### Delta-Gamma Hedging

50,000-path Monte Carlo comparing delta-only vs delta-gamma hedging (S0 = K1 = 100, K2 = 110, T = 1y, daily rebalancing):

| Strategy | Mean P&L | Std P&L | Std Reduction | 95% Range |
|---|---:|---:|---:|---|
| Delta only | ~$0.00 | $0.44 | --- | [-0.92, +0.90] |
| Delta-gamma | ~$0.00 | $0.24 | **46%** | [-0.35, +0.32] |

### P&L Variance Attribution

| Source | Correct Vol | Vol Misspecified (25% vs 20%) |
|---|---:|---:|
| Discrete Rebalancing | 97% | 22% |
| Vol Misspecification | 0% | 68% |
| Higher-Order Terms | 3% | 10% |

## Project Structure

```
options-vol-surface/
|-- src/                         # Core library (12 modules)
|   |-- __init__.py              # Public API re-exports
|   |-- config.py                # Constants: RANDOM_SEED, DEFAULT_N_MC_PATHS, etc.
|   |-- pricing.py               # BlackScholes class, binomial tree, Monte Carlo engine
|   |-- greeks.py                # Analytical + numerical Greeks (10 sensitivities)
|   |-- implied_vol.py           # Newton-Raphson IV solver with bisection fallback
|   |-- vol_surface.py           # SVI calibration, arbitrage checks, interpolation
|   |-- realized_vol.py          # Close-to-close, Parkinson, Yang-Zhang estimators
|   |-- delta_hedge.py           # MC delta-hedge simulation + single-path detail
|   |-- enhanced_hedging.py      # Delta-gamma hedging, P&L attribution, vega hedging
|   |-- historical_case_studies.py  # COVID crash, earnings IV crush, flash crash
|   |-- model_risk_analysis.py   # Model risk report, Monte Carlo precision analysis
|   |-- performance_benchmarks.py   # Timing + memory benchmarks for all pricing methods
|   |-- data_utils.py            # SPY options fetch (yfinance), FRED rates, cleaning
|-- tests/                       # 80 unit tests
|   |-- test_pricing.py          # Put-call parity, binomial convergence, MC CI
|   |-- test_greeks.py           # 42 parametrized: 7 Greeks x 6 variants (call/put x ITM/ATM/OTM)
|   |-- test_implied_vol.py      # IV recovery, extremes, bad price handling
|   |-- test_vol_surface.py      # SVI fit, butterfly arb, calendar arb, positive variance
|   |-- test_enhancements.py     # Delta-gamma, P&L attribution, case studies, benchmarks
|-- notebooks/                   # Interactive analysis (4 notebooks)
|   |-- 01_pricing_demo.ipynb    # Pricing comparison, MC convergence (3 figures, 2 tables)
|   |-- 02_greeks_analysis.ipynb # Greek surfaces, sensitivities (7 figures, 1 table)
|   |-- 03_vol_surface.ipynb     # SVI calibration, smile dynamics (7 figures, 2 tables)
|   |-- 04_delta_hedging.ipynb   # Hedging P&L, gamma-theta decomp (6 figures, 3 tables)
|-- report/                      # LaTeX academic report
|   |-- main.tex                 # 6 sections + 3 appendices, 42 equations, 20 figures
|   |-- references.bib           # 18 BibTeX entries
|   |-- figures/                 # Figures for LaTeX compilation
|-- data/raw/                    # Cached SPY options data (Parquet)
|-- results/figures/             # Generated plots from notebooks
|-- results/tables/              # Generated CSV/LaTeX tables
|-- requirements.txt
```

## Source Code Guide

### Core Pricing (`src/pricing.py`)
`BlackScholes` class with 6 parameters (S, K, T, r, q, sigma). Three pricing engines: analytical Black-Scholes-Merton, Cox-Ross-Rubinstein binomial tree (European and American), and Monte Carlo with antithetic variates and control variate variance reduction. All methods are vectorized with NumPy.

### Greeks (`src/greeks.py`)
`NumericalGreeks` class with 7 static methods computing finite-difference Greeks via central differencing. `compare_greeks()` validates numerical vs analytical values. `greek_surface_data()` generates delta/gamma/vega surfaces across spot and expiry grids. Supports all 10 Greeks: delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color.

### Implied Volatility (`src/implied_vol.py`)
Newton-Raphson solver with Brenner-Subrahmanyam (1988) initial seed for fast convergence. Automatic bisection fallback when vega is near zero (deep ITM/OTM). Returns `NaN` for arbitrage-violating prices.

### Volatility Surface (`src/vol_surface.py`)
Gatheral's SVI raw parameterization with two-stage calibration: differential evolution global search + L-BFGS-B local refinement. Includes butterfly arbitrage check (risk-neutral density non-negativity) and calendar spread arbitrage check (total variance monotonicity). Multi-seed retry for robustness.

### Realized Volatility (`src/realized_vol.py`)
Three estimators: close-to-close (standard), Parkinson high-low range (5x more efficient), and Yang-Zhang OHLC (combines overnight jumps with intraday range). All expect pandas Series input with configurable rolling windows.

### Delta Hedging (`src/delta_hedge.py`)
`run_delta_hedge()` runs 10,000+ path Monte Carlo with discrete daily rebalancing and configurable transaction costs. `single_path_detail()` provides step-by-step gamma-theta P&L decomposition for a single path. Premium-in-cash self-financing convention.

### Enhanced Hedging (`src/enhanced_hedging.py`)
Vectorized `black_scholes_greeks()` accepting array inputs. `delta_gamma_hedge_simulation()` compares delta-only vs delta-gamma hedging by adding a second option to neutralize gamma. `pnl_attribution_analysis()` decomposes hedge P&L variance into discrete rebalancing error, volatility misspecification, and higher-order terms. `vega_hedge_simulation()` models impact of vol shocks.

### Historical Case Studies (`src/historical_case_studies.py`)
Three case studies grounding theoretical results in market reality: COVID-19 crash (March 2020, SPX -34%, VIX 82.7), earnings IV crush (NVDA-style, 65% to 38% overnight), and the Flash Crash (May 2010, 9% drop in 5 minutes). Each includes spot/IV data, P&L breakdown, and key lessons.

### Model Risk Analysis (`src/model_risk_analysis.py`)
`generate_model_risk_report()` catalogs 7 model risk items with typical and stress impact in basis points. `analyze_monte_carlo_risk()` computes confidence intervals and relative standard errors for MC pricing.

### Performance Benchmarks (`src/performance_benchmarks.py`)
Timing and memory profiling for BSM (<1ms), binomial tree (19ms at N=1000), and Monte Carlo (67ms at 500K paths). Results exported as a DataFrame for the report appendix.

### Configuration (`src/config.py`)
Central constants: `RANDOM_SEED=42`, `DEFAULT_N_MC_PATHS=500_000`, default market parameters, and file paths. All notebooks and simulations reference this for reproducibility.

### Data Utilities (`src/data_utils.py`)
Fetches SPY options chains via yfinance with a 7-step cleaning pipeline: removes zero bids, filters extreme moneyness, eliminates low liquidity, checks bid-ask spreads, and caches to Parquet. Optional FRED API integration for live Treasury rates.

## Methodology

- **Pricing**: Black-Scholes analytical, CRR binomial (N=1,000), Monte Carlo (500K paths) with antithetic + control variate variance reduction
- **Greeks**: 10 analytical Greeks + numerical finite differences (central differencing, relative error < 0.1%)
- **IV Solver**: Newton-Raphson with Brenner-Subrahmanyam seed, bisection fallback
- **Vol Surface**: SVI raw parameterization, DE + L-BFGS-B two-stage calibration
- **Arbitrage**: Butterfly (density non-negativity) and calendar spread (total variance monotonicity)
- **Realized Vol**: Close-to-close, Parkinson, Yang-Zhang with rolling windows
- **Hedging**: 50,000-path MC, daily rebalancing, configurable transaction costs, gamma-theta decomposition
- **Delta-Gamma**: Second option neutralizes gamma via n2 = Gamma1/Gamma2, 46% std reduction
- **P&L Attribution**: 3-component variance decomposition (discrete rebalancing, vol misspec, higher-order)

## Installation

```bash
git clone https://github.com/zaid282802/options-vol-surface.git
cd options-vol-surface
pip install -r requirements.txt
```

**Optional**: Set `FRED_API_KEY` as an environment variable to fetch live Treasury rates from
FRED. Without it, the system falls back to the configured rate (4.3%).

## Usage

```bash
# Run all 80 tests
pytest tests/ -v

# Launch analysis notebooks
jupyter notebook notebooks/
```

## Test Coverage

| Test File | Tests | What It Covers |
|---|---:|---|
| test_pricing.py | 15 | Put-call parity, binomial convergence, MC CI, deep moneyness |
| test_greeks.py | 42 | 7 Greeks x 6 variants (call/put x ITM/ATM/OTM), REL_TOL=0.001 |
| test_implied_vol.py | 6 | IV recovery, extreme vols, bad price returns NaN |
| test_vol_surface.py | 5 | SVI flat vol, fit quality, butterfly arb, calendar arb |
| test_enhancements.py | 12 | Delta-gamma hedge, P&L attribution, case studies, benchmarks |
| **Total** | **80** | |


## References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229-263.
- Brenner, M. & Subrahmanyam, M. (1988). A Simple Formula to Compute the Implied Standard Deviation. *The Journal of Finance*, 43(4).
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI Volatility Surfaces. *Quantitative Finance*, 14(1), 59-71.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- El Karoui, N., Jeanblanc-Picque, M. & Shreve, S. (1998). Robustness of the Black and Scholes Formula. *Mathematical Finance*, 8(2), 93-126.
- Hull, J. (2021). *Options, Futures, and Other Derivatives*. 11th Ed. Pearson.
- Sinclair, E. (2010). *Option Trading: Pricing and Volatility Strategies and Techniques*. Wiley.

## Author

**Zaid Annigeri** -- Master of Quantitative Finance, Rutgers Business School
[LinkedIn](https://linkedin.com/in/zed228) | [GitHub](https://github.com/zaid282802)

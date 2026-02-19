"""Performance benchmarks: timing, memory, scalability for all pricing components."""

import numpy as np
import time
import tracemalloc
from typing import List, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
from scipy.stats import norm


@dataclass
class BenchmarkResult:
    operation: str
    time_ms: float
    memory_mb: float
    parallelizable: bool
    complexity: str
    notes: str


def time_function(func, *args, n_runs=10, **kwargs):
    """Time a function over multiple runs. Returns (mean_ms, std_ms)."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def measure_memory(func, *args, **kwargs):
    """Measure peak memory in MB."""
    tracemalloc.start()
    func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def bs_call(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def binomial_call(S, K, T, r, q, sigma, N=1000):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))
    payoffs = np.maximum(ST - K, 0)

    for i in range(N - 1, -1, -1):
        payoffs = np.exp(-r * dt) * (p * payoffs[1:i + 2] + (1 - p) * payoffs[0:i + 1])
    return payoffs[0]


def mc_call(S, K, T, r, q, sigma, n_paths=100000):
    Z = np.random.standard_normal(n_paths // 2)
    Z = np.concatenate([Z, -Z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))


def run_benchmarks():
    """Run all benchmarks and return list of BenchmarkResult."""
    results = []
    S, K, T, r, q, sigma = 100, 100, 0.25, 0.05, 0.02, 0.20

    t, _ = time_function(bs_call, S, K, T, r, q, sigma, n_runs=100)
    m = measure_memory(bs_call, S, K, T, r, q, sigma)
    results.append(BenchmarkResult("BSM (1 option)", t, m, True, "O(1)", "Analytical"))

    for N in [100, 500, 1000]:
        t, _ = time_function(binomial_call, S, K, T, r, q, sigma, N, n_runs=10)
        m = measure_memory(binomial_call, S, K, T, r, q, sigma, N)
        results.append(BenchmarkResult(f"Binomial (N={N})", t, m, False, "O(N)", "Error ~O(1/N)"))

    for paths in [10000, 100000, 500000]:
        t, _ = time_function(mc_call, S, K, T, r, q, sigma, paths, n_runs=5)
        m = measure_memory(mc_call, S, K, T, r, q, sigma, paths)
        results.append(BenchmarkResult(f"MC ({paths // 1000}K)", t, m, True, "O(N)", "SE ~O(1/sqrt(N))"))

    return results


def results_to_dataframe(results):
    return pd.DataFrame([
        {
            'Operation': r.operation,
            'Time (ms)': f"{r.time_ms:.2f}",
            'Memory (MB)': f"{r.memory_mb:.2f}",
            'Parallel': 'Yes' if r.parallelizable else 'No',
            'Complexity': r.complexity,
        }
        for r in results
    ])


if __name__ == "__main__":
    print("Running benchmarks...")
    results = run_benchmarks()
    print()
    print("PERFORMANCE BENCHMARKS")
    print(results_to_dataframe(results).to_string(index=False))

"""
attrition_toy_simulation.py — Pure-function Monte Carlo toy simulator.

ISOLATION NOTICE
================
This module is intentionally disconnected from the XGBoost / Cox PH survival
pipeline, the OLAP warehouse, Neo4j, and every other production subsystem.
It has ZERO callers outside server.py (the API adapter).

Do NOT import this from production_ml.py, simulator_actions.py, etl.py,
or any module in the ML training/inference path.
================

It draws per-month attrition rates from a Beta distribution, rescaled to a
configurable min and max range. It then compounds these rates over a fixed 12-month
horizon to compute monthly headcount trajectories.
"""

from typing import Dict, List

import numpy as np

# ── Locked parameters ────────────────────────────────────────────────────────
DEFAULT_ALPHA: float = 2.0
DEFAULT_BETA: float = 2.0
DEFAULT_N_SIMS: int = 10_000
DEFAULT_STARTING_HEADCOUNT: int = 1489
MAX_N_SIMS: int = 100_000
_MONTHS: int = 12
DEFAULT_RATE_LO: float = 0.05  # 5%
DEFAULT_RATE_HI: float = 0.15  # 15%


def simulate(
    n_sims: int = DEFAULT_N_SIMS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    starting_headcount: int = DEFAULT_STARTING_HEADCOUNT,
    rate_lo: float = DEFAULT_RATE_LO,
    rate_hi: float = DEFAULT_RATE_HI,
    seed: int | None = None,
) -> Dict:
    """Run a toy attrition headcount simulation.

    Parameters
    ----------
    n_sims : int
        Number of independent simulations. Must be in [1, MAX_N_SIMS].
    alpha, beta : float
        Shape parameters for the Beta distribution (both must be > 0).
    starting_headcount : int
        Initial headcount at month 0.
    rate_lo, rate_hi : float
        The minimum and maximum possible monthly attrition rates (as decimals).
    seed : int or None
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        scenarios : list[dict]
            List of length n_sims. Each dict contains:
                id (int): Scenario ID
                rates (list[float]): 12 monthly rates (as decimals, e.g., 0.10 for 10%)
                headcounts (list[int]): 13 headcount values (month 0 to 12)
                final_headcount (int): Headcount at month 12
                total_lost (int): starting_headcount - final_headcount
        stats : dict
            Summary statistics of the final headcount (mean, p10, p50, p90, min, max).
    """
    # ── Validation ────────────────────────────────────────────────────────
    if not isinstance(n_sims, int) or n_sims < 1 or n_sims > MAX_N_SIMS:
        raise ValueError(f"n_sims must be an integer in [1, {MAX_N_SIMS}], got {n_sims}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if starting_headcount < 1:
        raise ValueError(f"starting_headcount must be >= 1, got {starting_headcount}")
    if rate_lo < 0 or rate_hi > 1 or rate_lo >= rate_hi:
        raise ValueError("Invalid rate_lo/rate_hi bounds. Must be 0 <= rate_lo < rate_hi <= 1")

    # ── Simulation ────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # Draw raw Beta samples, shape (n_sims, 12), and rescale to [rate_lo, rate_hi]
    rates = rate_lo + rng.beta(alpha, beta, size=(n_sims, _MONTHS)) * (rate_hi - rate_lo)

    # Compute headcounts sequentially to allow for integer rounding at each step
    headcounts = np.zeros((n_sims, _MONTHS + 1), dtype=np.int32)
    headcounts[:, 0] = starting_headcount

    for m in range(_MONTHS):
        # We subtract the people lost in month m
        lost = np.round(headcounts[:, m] * rates[:, m]).astype(np.int32)
        headcounts[:, m + 1] = headcounts[:, m] - lost

    final_headcounts = headcounts[:, -1]
    
    # ── Prepare Output ────────────────────────────────────────────────────
    
    # Pre-allocate the list of dicts for performance
    scenarios = [None] * n_sims
    
    # Convert numpy arrays to standard Python types before returning to avoid JSON serialization issues
    rates_list = rates.tolist()
    headcounts_list = headcounts.tolist()
    
    for i in range(n_sims):
        h = headcounts_list[i]
        final = h[-1]
        scenarios[i] = {
            "id": i + 1,
            "rates": rates_list[i],
            "headcounts": h,
            "final_headcount": final,
            "total_lost": starting_headcount - final
        }

    return {
        "scenarios": scenarios,
        "stats": {
            "mean": float(np.mean(final_headcounts)),
            "p10": float(np.percentile(final_headcounts, 10)),
            "p50": float(np.percentile(final_headcounts, 50)),
            "p90": float(np.percentile(final_headcounts, 90)),
            "min": int(np.min(final_headcounts)),
            "max": int(np.max(final_headcounts)),
        },
    }


if __name__ == "__main__":
    result = simulate(n_sims=10_000, seed=42, starting_headcount=1489)
    s = result["stats"]
    print(f"final_headcount_stats: {{mean: {s['mean']:.2f}, p10: {s['p10']:.1f}, p50: {s['p50']:.1f}, p90: {s['p90']:.1f}, min: {float(s['min'])}, max: {float(s['max'])}}}")
    sample_h = result['scenarios'][0]['headcounts']
    print(f"sample scenario[0]: headcount_by_month starts {sample_h[:3]}... ends at {sample_h[-1]}")

"""Shared peptide knockoff Monte Carlo helpers."""

from collections import Counter
from typing import Dict, Optional, Tuple, Union

import numpy as np


def _mc_sum_from_pool(
    pool: Optional[np.ndarray],
    K: int,
    c: int,
    rng: np.random.Generator,
    sample_block_size: int,
) -> np.ndarray:
    """Sample K null sums of c shared peptide contributions from one pool."""
    if c <= 0:
        return np.zeros(int(K), dtype=np.float64)
    if pool is None or pool.size == 0:
        return np.zeros(int(K), dtype=np.float64)

    K = int(K)
    c = int(c)
    n = int(pool.size)
    replace = bool(c > n)
    block = int(max(1, sample_block_size))
    out = np.zeros(K, dtype=np.float64)
    i = 0
    while i < K:
        j = min(K, i + block)
        block_size = int(j - i)
        if replace:
            idx = rng.integers(0, n, size=(block_size, c), endpoint=False)
        else:
            idx = np.empty((block_size, c), dtype=np.int64)
            for r in range(block_size):
                idx[r, :] = rng.choice(n, size=c, replace=False)
        out[i:j] = pool[idx].sum(axis=1)
        i = j
    return out


def shared_knockoff_mc(
    gid: str,
    obs_shared_score: float,
    K: int,
    rng: np.random.Generator,
    pools: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]],
    counts_by_genome: Dict[str, Counter],
    sample_block_size: int,
) -> Tuple[float, float, float, float, float]:
    """Empirical p-value and null moments for shared evidence by knockoff MC."""
    counts = counts_by_genome.get(gid, None)
    if not counts:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    null_sum = np.zeros(int(K), dtype=np.float64)
    for key, c in counts.items():
        pool = pools.get(key, None) if pools else None
        null_sum += _mc_sum_from_pool(
            pool=pool,
            K=int(K),
            c=int(c),
            rng=rng,
            sample_block_size=sample_block_size,
        )

    ge = float(np.sum(null_sum >= float(obs_shared_score)))
    p = (1.0 + ge) / (1.0 + float(K))
    mu = float(null_sum.mean())
    sd = float(null_sum.std(ddof=1)) if int(K) > 1 else 0.0
    p95 = float(np.quantile(null_sum, 0.95))
    p99 = float(np.quantile(null_sum, 0.99))
    return float(p), mu, sd, p95, p99

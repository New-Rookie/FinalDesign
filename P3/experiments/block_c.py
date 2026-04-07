"""
Experiment Block C — Average total energy under resource variation.

This block's data is produced by block_b.py (the merged B+C run).
This module provides a standalone entry point that delegates to run_block_bc
and returns only the energy summary.

Output: block_c_raw.csv, block_c_summary.csv  (written by block_b.run_block_bc)
"""

from __future__ import annotations

import os
import time

import pandas as pd

from P3.experiments.block_b import run_block_bc


def run_block_c(
    log_dir: str = "P3/logs",
    **kwargs,
) -> pd.DataFrame:
    """Run block C by delegating to the merged B+C runner."""
    c_path = os.path.join(log_dir, "block_c_summary.csv")
    if os.path.exists(c_path):
        return pd.read_csv(c_path)
    _, sum_c = run_block_bc(log_dir=log_dir, **kwargs)
    return sum_c


if __name__ == "__main__":
    t0 = time.time()
    s = run_block_c()
    print(s.to_string(index=False))
    print(f"\nBlock C completed in {time.time() - t0:.1f}s")

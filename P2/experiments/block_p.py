"""
Experiment Block P — Offline Random Forest estimator stage.

Collect labelled probe samples, train per-class RF regressors,
calibrate with isotonic regression, validate with blocked K-fold.

Output: block_p_raw.csv, block_p_summary.csv
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from P2.link_quality.probe_collector import collect_probes
from P2.link_quality.rf_estimator import (
    LinkQualityEstimator, LINK_CLASSES,
)
from utils.result_saver import save_block_results

N_PROBE_PER_CLASS = 20_000
N_STEPS = 600


def run_block_p(log_dir: str = "P2/logs",
                n_probe: int = N_PROBE_PER_CLASS,
                n_steps: int = N_STEPS,
                seed: int = 0) -> Dict:
    """Train and validate RF estimator, return validation artifacts."""
    os.makedirs(log_dir, exist_ok=True)

    print("  [Block P] Collecting probe samples ...")
    cfg = EnvConfig(N_total=20, eta_ch=1.0, print_diagnostics=False)
    df_probes = collect_probes(cfg, n_steps=n_steps,
                               n_probe_per_class=n_probe, seed=seed)

    print(f"  [Block P] Collected {len(df_probes)} samples across "
          f"{df_probes['link_class'].nunique()} classes")
    for lc in LINK_CLASSES:
        cnt = len(df_probes[df_probes["link_class"] == lc])
        print(f"    {lc:20s}: {cnt}")

    print("  [Block P] Training RF estimators ...")
    estimator = LinkQualityEstimator(n_estimators=100, max_depth=12,
                                     random_state=seed)
    val_metrics = estimator.train(df_probes)

    raw_records = []
    summary_records = []
    for lc in LINK_CLASSES:
        m = val_metrics.get(lc, {})
        r2 = m.get("R2", 0.0)
        mae = m.get("MAE", 1.0)
        n_samp = m.get("n_samples", 0)
        summary_records.append({
            "link_class": lc, "R2": r2, "MAE": mae, "n_samples": n_samp,
        })
        pred = m.get("pred")
        true = m.get("true")
        if pred is not None and true is not None:
            for p_val, t_val in zip(pred, true):
                raw_records.append({
                    "link_class": lc,
                    "prr_pred": float(p_val),
                    "prr_true": float(t_val),
                })
        print(f"    {lc:20s}: R²={r2:.4f}  MAE={mae:.4f}  n={n_samp}")

    df_raw = pd.DataFrame(raw_records)
    df_summary = pd.DataFrame(summary_records)

    save_block_results("P2", "P", raw_df=df_raw, summary_df=df_summary, log_dir=log_dir)

    model_path = os.path.join(log_dir, "rf_estimator.pkl")
    estimator.save(model_path)
    print(f"  [Block P] Model saved to {model_path}")

    return {
        "estimator": estimator,
        "summary": df_summary,
        "raw": df_raw,
        "model_path": model_path,
    }


if __name__ == "__main__":
    t0 = time.time()
    result = run_block_p()
    print(result["summary"].to_string(index=False))
    print(f"\nBlock P completed in {time.time() - t0:.1f}s")

"""
Master experiment runner — orchestrates all six experiment blocks
for Research Content 1 (INDP neighbour discovery).

Usage:
    python -m P1.experiments.runner [--blocks A B C D E F] [--quick] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from P1.experiments.block_a import run_block_a
from P1.experiments.block_b import run_block_b
from P1.experiments.block_c import run_block_c
from P1.experiments.block_d import run_block_d
from P1.experiments.block_e import run_block_e
from P1.experiments.block_f import run_block_f
from utils.result_saver import get_result_dir


BLOCK_MAP = {
    "A": run_block_a,
    "B": run_block_b,
    "C": run_block_c,
    "D": run_block_d,
    "E": run_block_e,
    "F": run_block_f,
}

BLOCK_DESC = {
    "A": "Mechanism comparison vs noise (F1_topo)",
    "B": "Mechanism comparison vs node count (F1_topo)",
    "C": "Actor/Critic LR pair sweep (ME-IPPO reward curve)",
    "D": "Algorithm comparison vs noise (total E_ND)",
    "E": "Algorithm comparison vs node count (total E_ND)",
    "F": "Convergence comparison (ME-IPPO vs Improved IPPO vs IPPO)",
}


def _resolve_execution_config(device_arg: str, cpu_cores: int | None, cpu_utilization: float) -> tuple[str, int, int]:
    cpu_total = os.cpu_count() or 1
    cpu_budget = max(1, min(cpu_total, cpu_cores if cpu_cores is not None else int(round(cpu_total * cpu_utilization))))

    device = "cpu"
    if device_arg in ("cpu", "cuda"):
        device = device_arg
    else:
        import importlib.util
        import importlib
        has_torch = importlib.util.find_spec("torch") is not None
        if has_torch:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                device = "cuda"

    worker_budget = cpu_budget

    if device == "cuda":
        worker_budget = max(1, min(worker_budget, 4))

    return device, cpu_budget, worker_budget


def main():
    default_log_dir = get_result_dir("P1")

    parser = argparse.ArgumentParser(description="RC1 INDP Experiment Runner")
    parser.add_argument("--blocks", nargs="*", default=list(BLOCK_MAP.keys()),
                        help="Blocks to run (default: all)")
    parser.add_argument("--log-dir", default=default_log_dir,
                        help="Output directory for CSV results")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce seeds/episodes for a fast smoke test")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (overrides auto budget)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Execution device preference for RL blocks")
    parser.add_argument("--cpu-cores", type=int, default=None,
                        help="CPU cores budget for this runner")
    parser.add_argument("--cpu-utilization", type=float, default=1.0,
                        help="Fraction of local CPU to budget when --cpu-cores is not set")
    parser.add_argument("--rl-episodes", type=int, default=None,
                        help="Override RL training episodes for episodic RL blocks")
    parser.add_argument("--rl-windows", type=int, default=10,
                        help="Training timesteps/windows per RL episode")
    args = parser.parse_args()

    device, cpu_budget, auto_workers = _resolve_execution_config(
        args.device, args.cpu_cores, args.cpu_utilization)
    resolved_workers = args.workers if args.workers is not None else auto_workers

    quick_kw = {}
    if args.quick:
        quick_kw = {"n_seeds": 2}

    worker_kw = {"n_workers": resolved_workers}

    os.makedirs(args.log_dir, exist_ok=True)

    blocks_to_run = [b.upper() for b in args.blocks if b.upper() in BLOCK_MAP]
    if not blocks_to_run:
        print("No valid blocks specified.")
        return

    print()
    print("=" * 64)
    print("  Research Content 1 — INDP Experiment Campaign")
    print("=" * 64)
    if args.quick:
        print("  MODE: --quick (reduced seeds/episodes for smoke test)")
    print(f"  Blocks: {', '.join(blocks_to_run)}")
    print(f"  Result dir: {args.log_dir}/")
    print(f"  Device: {device}")
    print(f"  CPU budget: {cpu_budget} cores")
    print(f"  Workers: {resolved_workers} ({'manual' if args.workers is not None else 'auto'})")
    print("=" * 64)
    print()

    total_t0 = time.time()

    pbar_blocks = tqdm(blocks_to_run, desc="Overall P1", unit="block",
                       leave=True, dynamic_ncols=True)

    for block_id in pbar_blocks:
        desc = BLOCK_DESC.get(block_id, "")
        pbar_blocks.set_postfix_str(f"Block {block_id}: {desc}")

        print(f"\n{'─' * 64}")
        print(f"  Block {block_id} — {desc}")
        print(f"{'─' * 64}")
        t0 = time.time()

        if block_id == "C":
            kw = {**quick_kw, **worker_kw, "device": device}
            if args.quick:
                kw["n_episodes"] = 10
                kw["n_windows"] = 3
                kw["n_seeds"] = 2
            if args.rl_episodes is not None:
                kw["n_episodes"] = args.rl_episodes
            kw["n_windows"] = args.rl_windows
            BLOCK_MAP[block_id](log_dir=args.log_dir, **kw)
        elif block_id in ("D", "E"):
            kw = {**quick_kw, **worker_kw, "device": device}
            if args.quick:
                kw["n_train"] = 10
                kw["n_windows_train"] = 3
            if args.rl_episodes is not None:
                kw["n_train"] = args.rl_episodes
            kw["n_windows_train"] = args.rl_windows
            BLOCK_MAP[block_id](log_dir=args.log_dir, **kw)
        elif block_id == "F":
            kw = {**worker_kw, "device": device}
            if args.quick:
                kw["n_episodes"] = 10
                kw["n_windows"] = 3
                kw["n_seeds"] = 2
            if args.rl_episodes is not None:
                kw["n_episodes"] = args.rl_episodes
            kw["n_windows"] = args.rl_windows
            BLOCK_MAP[block_id](log_dir=args.log_dir, **kw)
        else:
            BLOCK_MAP[block_id](log_dir=args.log_dir, **quick_kw, **worker_kw)

        elapsed = time.time() - t0
        print(f"  Block {block_id} completed in {elapsed:.1f}s")

    pbar_blocks.close()

    total = time.time() - total_t0
    print(f"\n{'=' * 64}")
    print(f"  All P1 blocks completed in {total:.1f}s")
    print(f"  Results saved to: {args.log_dir}/")
    print(f"{'=' * 64}")

    try:
        from P1.plotting.plot_results import generate_all_figures
        generate_all_figures(log_dir=args.log_dir, fig_dir="P1/figures")
        print("  Figures saved to: P1/figures/")
    except Exception as e:
        print(f"  Plot generation failed: {e}")


if __name__ == "__main__":
    main()

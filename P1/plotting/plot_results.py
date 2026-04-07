"""
Plot generation for Research Content 1 experiments.

Produces 6 figures as specified in Experiment Manual I:
  Fig1 — F1_topo vs eta_N (mechanism comparison)
  Fig2 — F1_topo vs N_total (mechanism comparison)
  Fig3 — Reward curve vs training episode (ME-IPPO lr sweep)
  Fig4 — E_ND vs eta_N (algorithm comparison)
  Fig5 — E_ND vs N_total (algorithm comparison)
  Fig6 — Convergence comparison across algorithms
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

MECH_MARKERS = {"INDP": "o", "Disco": "s", "ALOHA": "^"}
MECH_COLORS = {"INDP": "#2196F3", "Disco": "#FF9800", "ALOHA": "#4CAF50"}

ALGO_MARKERS = {"ME_IPPO": "P", "Improved_IPPO": "o", "IPPO": "s",
                "Greedy": "^", "ACO": "D", "GA": "v"}
ALGO_COLORS = {"ME_IPPO": "#D32F2F", "Improved_IPPO": "#E91E63",
               "IPPO": "#2196F3", "Greedy": "#FF9800",
               "ACO": "#4CAF50", "GA": "#9C27B0"}


def _safe_load(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — F1 vs noise (mechanism)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_a_summary.csv"))
    if df is None:
        print("  [Fig1] block_a_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for mech in ["INDP", "Disco", "ALOHA"]:
        col_mean = f"{mech}_mean"
        col_std = f"{mech}_std" if f"{mech}_std" in df.columns else None
        if col_mean not in df.columns:
            continue
        sub = df.sort_values("eta_N")
        yerr = sub[col_std] if col_std and col_std in df.columns else None
        ax.errorbar(sub["eta_N"], sub[col_mean], yerr=yerr,
                    marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                    label=mech, capsize=3, linewidth=1.5)
    ax.set_xlabel("Noise scale $\\eta_N$")
    ax.set_ylabel("$F1_{topo}$")
    ax.set_title("Mechanism Comparison — Accuracy vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig1_RC1_accuracy_vs_noise.png"))
    plt.close(fig)
    print("  [Fig1] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — F1 vs N_total (mechanism)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig2(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_b_summary.csv"))
    if df is None:
        print("  [Fig2] block_b_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for mech in ["INDP", "Disco", "ALOHA"]:
        col_mean = f"{mech}_mean"
        col_std = f"{mech}_std" if f"{mech}_std" in df.columns else None
        if col_mean not in df.columns:
            continue
        sub = df.sort_values("N_total")
        yerr = sub[col_std] if col_std and col_std in df.columns else None
        ax.errorbar(sub["N_total"], sub[col_mean], yerr=yerr,
                    marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                    label=mech, capsize=3, linewidth=1.5)
    ax.set_xlabel("Total node count $N_{total}$")
    ax.set_ylabel("$F1_{topo}$")
    ax.set_title("Mechanism Comparison — Accuracy vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig2_RC1_accuracy_vs_nodes.png"))
    plt.close(fig)
    print("  [Fig2] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Reward curve (lr sweep)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig3(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_c_summary.csv"))
    if df is None:
        print("  [Fig3] block_c_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for lr_label in sorted(df["lr_label"].unique()):
        sub = df[df["lr_label"] == lr_label].sort_values("episode")
        ax.plot(sub["episode"], sub["mean"], label=lr_label, linewidth=1.5)
        if "std" in sub.columns:
            ax.fill_between(sub["episode"],
                            sub["mean"] - sub["std"].fillna(0),
                            sub["mean"] + sub["std"].fillna(0),
                            alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("ME-IPPO — Reward Curve (Learning Rate Sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig3_RC1_reward_lr.png"))
    plt.close(fig)
    print("  [Fig3] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — E_ND vs noise (algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig4(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_d_summary.csv"))
    if df is None:
        print("  [Fig4] block_d_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["ME_IPPO", "Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]:
        col_mean = f"{algo}_mean"
        col_std = f"{algo}_std" if f"{algo}_std" in df.columns else None
        if col_mean not in df.columns:
            continue
        sub = df.sort_values("eta_N")
        yerr = sub[col_std] if col_std and col_std in df.columns else None
        ax.errorbar(sub["eta_N"], sub[col_mean], yerr=yerr,
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=algo.replace("_", " "), capsize=3, linewidth=1.5)
    ax.set_xlabel("Noise scale $\\eta_N$")
    ax.set_ylabel("Total $E_{ND}$ (J)")
    ax.set_title("Algorithm Comparison — Energy vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig4_RC1_energy_vs_noise.png"))
    plt.close(fig)
    print("  [Fig4] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — E_ND vs N_total (algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig5(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_e_summary.csv"))
    if df is None:
        print("  [Fig5] block_e_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["ME_IPPO", "Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]:
        col_mean = f"{algo}_mean"
        col_std = f"{algo}_std" if f"{algo}_std" in df.columns else None
        if col_mean not in df.columns:
            continue
        sub = df.sort_values("N_total")
        yerr = sub[col_std] if col_std and col_std in df.columns else None
        ax.errorbar(sub["N_total"], sub[col_mean], yerr=yerr,
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=algo.replace("_", " "), capsize=3, linewidth=1.5)
    ax.set_xlabel("Total node count $N_{total}$")
    ax.set_ylabel("Total $E_{ND}$ (J)")
    ax.set_title("Algorithm Comparison — Energy vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig5_RC1_energy_vs_nodes.png"))
    plt.close(fig)
    print("  [Fig5] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — convergence comparison (algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig6(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_f_summary.csv"))
    if df is None:
        print("  [Fig6] block_f_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["ME_IPPO", "Improved_IPPO", "IPPO"]:
        sub = df[df["algorithm"] == algo].sort_values("episode")
        if sub.empty:
            continue
        ax.plot(sub["episode"], sub["mean"],
                marker=ALGO_MARKERS.get(algo, "o"),
                color=ALGO_COLORS.get(algo, "gray"),
                label=algo.replace("_", " "), linewidth=1.5,
                markevery=max(1, len(sub) // 20))
        if "std" in sub.columns:
            ax.fill_between(sub["episode"],
                            sub["mean"] - sub["std"].fillna(0),
                            sub["mean"] + sub["std"].fillna(0),
                            alpha=0.15, color=ALGO_COLORS.get(algo, "gray"))
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("Convergence Comparison — RC1 Algorithms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig6_RC1_convergence_algorithms.png"))
    plt.close(fig)
    print("  [Fig6] saved.")

# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(log_dir: str = "P1/logs",
                         fig_dir: str = "P1/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating Research Content 1 figures...")
    plot_fig1(log_dir, fig_dir)
    plot_fig2(log_dir, fig_dir)
    plot_fig3(log_dir, fig_dir)
    plot_fig4(log_dir, fig_dir)
    plot_fig5(log_dir, fig_dir)
    plot_fig6(log_dir, fig_dir)
    print("Done.\n")


if __name__ == "__main__":
    generate_all_figures()

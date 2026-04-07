"""
Plot generation for Research Content 3 experiments.

Produces 9 figures:
  Fig1  — Improved MATD3 reward curve under learning-rate sweep
  Fig2  — Average T_total vs eta_B  (algorithm comparison)
  Fig3  — Average T_total vs eta_F
  Fig4  — Average T_total vs eta_S
  Fig5  — Average E_total vs eta_B
  Fig6  — Average E_total vs eta_F
  Fig7  — Average E_total vs eta_S
  Fig8  — Average T_total vs M_tot
  Fig9  — Average E_total vs M_tot
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

ALGO_MARKERS = {
    "Improved_MATD3": "o", "MATD3": "s",
    "Greedy": "^", "ACO": "D", "GA": "v",
}
ALGO_COLORS = {
    "Improved_MATD3": "#E91E63", "MATD3": "#2196F3",
    "Greedy": "#FF9800", "ACO": "#4CAF50", "GA": "#9C27B0",
}
ALGO_LABELS = {
    "Improved_MATD3": "Improved MATD3", "MATD3": "MATD3",
    "Greedy": "Greedy", "ACO": "ACO", "GA": "GA",
}

LR_COLORS = {1e-4: "#2196F3", 3e-4: "#FF9800", 1e-3: "#4CAF50"}
LR_LABELS = {1e-4: "lr=1e-4", 3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}


def _safe_load(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — Reward curve (LR sweep)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_a_summary.csv"))
    if df is None:
        print("  [Fig1] block_a_summary.csv not found, skipping.")
        return
    fig, ax = plt.subplots()
    for lr_val in sorted(df["lr"].unique()):
        sub = df[df["lr"] == lr_val].sort_values("episode")
        color = LR_COLORS.get(lr_val, "gray")
        label = LR_LABELS.get(lr_val, f"lr={lr_val}")
        ax.plot(sub["episode"], sub["mean"], color=color,
                label=label, linewidth=1.5)
        ax.fill_between(sub["episode"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("Improved MATD3 — Reward Curve (Learning Rate Sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig1_RC3_reward_lr.png"))
    plt.close(fig)
    print("  [Fig1] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figs 2-4 — Average T_total vs eta (delay)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_eta_metric(
    log_dir, fig_dir, eta_type, metric_col,
    ylabel, title_suffix, fig_name,
):
    df = _safe_load(os.path.join(log_dir, "block_b_summary.csv" if "T_total" in metric_col
                                 else "block_c_summary.csv"))
    if df is None:
        summary_path = os.path.join(log_dir, "block_b_summary.csv")
        df = _safe_load(summary_path)
    if df is None:
        print(f"  [{fig_name}] summary CSV not found, skipping.")
        return

    sub_all = df[df["eta_type"] == eta_type]
    if sub_all.empty:
        print(f"  [{fig_name}] no data for {eta_type}, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]:
        sub = sub_all[sub_all["algorithm"] == algo].sort_values("eta_value")
        if sub.empty:
            continue
        ax.errorbar(sub["eta_value"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=ALGO_LABELS.get(algo, algo),
                    capsize=3, linewidth=1.5)
    eta_sym = {"eta_B": r"$\eta_B$", "eta_F": r"$\eta_F$",
               "eta_S": r"$\eta_S$"}.get(eta_type, eta_type)
    ax.set_xlabel(f"Resource scaling {eta_sym}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Algorithm Comparison — {title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close(fig)
    print(f"  [{fig_name}] saved.")


def plot_fig2(log_dir, fig_dir):
    _plot_eta_metric(log_dir, fig_dir, "eta_B", "mean_T_total",
                     "Average $T_{total}$ (s)",
                     "Delay vs Bandwidth Scaling",
                     "Fig2_RC3_delay_vs_etaB.png")

def plot_fig3(log_dir, fig_dir):
    _plot_eta_metric(log_dir, fig_dir, "eta_F", "mean_T_total",
                     "Average $T_{total}$ (s)",
                     "Delay vs Compute Scaling",
                     "Fig3_RC3_delay_vs_etaF.png")

def plot_fig4(log_dir, fig_dir):
    _plot_eta_metric(log_dir, fig_dir, "eta_S", "mean_T_total",
                     "Average $T_{total}$ (s)",
                     "Delay vs Storage Scaling",
                     "Fig4_RC3_delay_vs_etaS.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figs 5-7 — Average E_total vs eta (energy)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_eta_energy(
    log_dir, fig_dir, eta_type, fig_name, title_suffix,
):
    df = _safe_load(os.path.join(log_dir, "block_c_summary.csv"))
    if df is None:
        print(f"  [{fig_name}] block_c_summary.csv not found, skipping.")
        return

    sub_all = df[df["eta_type"] == eta_type]
    if sub_all.empty:
        print(f"  [{fig_name}] no data for {eta_type}, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]:
        sub = sub_all[sub_all["algorithm"] == algo].sort_values("eta_value")
        if sub.empty:
            continue
        ax.errorbar(sub["eta_value"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=ALGO_LABELS.get(algo, algo),
                    capsize=3, linewidth=1.5)
    eta_sym = {"eta_B": r"$\eta_B$", "eta_F": r"$\eta_F$",
               "eta_S": r"$\eta_S$"}.get(eta_type, eta_type)
    ax.set_xlabel(f"Resource scaling {eta_sym}")
    ax.set_ylabel("Average $E_{total}$ (J)")
    ax.set_title(f"Algorithm Comparison — {title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close(fig)
    print(f"  [{fig_name}] saved.")


def plot_fig5(log_dir, fig_dir):
    _plot_eta_energy(log_dir, fig_dir, "eta_B",
                     "Fig5_RC3_energy_vs_etaB.png",
                     "Energy vs Bandwidth Scaling")

def plot_fig6(log_dir, fig_dir):
    _plot_eta_energy(log_dir, fig_dir, "eta_F",
                     "Fig6_RC3_energy_vs_etaF.png",
                     "Energy vs Compute Scaling")

def plot_fig7(log_dir, fig_dir):
    _plot_eta_energy(log_dir, fig_dir, "eta_S",
                     "Fig7_RC3_energy_vs_etaS.png",
                     "Energy vs Storage Scaling")


# ═══════════════════════════════════════════════════════════════════════════
# Figs 8-9 — T_total and E_total vs M_tot
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig8(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_d_summary.csv"))
    if df is None:
        print("  [Fig8] block_d_summary.csv not found, skipping.")
        return
    fig, ax = plt.subplots()
    for algo in ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("M_tot_Mbit")
        if sub.empty:
            continue
        ax.errorbar(sub["M_tot_Mbit"], sub["mean_T"], yerr=sub["std_T"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=ALGO_LABELS.get(algo, algo),
                    capsize=3, linewidth=1.5)
    ax.set_xlabel("Total data volume $M_{tot}$ (Mbit)")
    ax.set_ylabel("Average $T_{total}$ (s)")
    ax.set_title("Algorithm Comparison — Delay vs Data Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig8_RC3_delay_vs_Mtot.png"))
    plt.close(fig)
    print("  [Fig8] saved.")


def plot_fig9(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_d_summary.csv"))
    if df is None:
        print("  [Fig9] block_d_summary.csv not found, skipping.")
        return
    fig, ax = plt.subplots()
    for algo in ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("M_tot_Mbit")
        if sub.empty:
            continue
        ax.errorbar(sub["M_tot_Mbit"], sub["mean_E"], yerr=sub["std_E"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=ALGO_LABELS.get(algo, algo),
                    capsize=3, linewidth=1.5)
    ax.set_xlabel("Total data volume $M_{tot}$ (Mbit)")
    ax.set_ylabel("Average $E_{total}$ (J)")
    ax.set_title("Algorithm Comparison — Energy vs Data Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig9_RC3_energy_vs_Mtot.png"))
    plt.close(fig)
    print("  [Fig9] saved.")




# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    log_dir: str = "P3/logs",
    fig_dir: str = "P3/figures",
):
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating Research Content 3 figures...")
    plot_fig1(log_dir, fig_dir)
    plot_fig2(log_dir, fig_dir)
    plot_fig3(log_dir, fig_dir)
    plot_fig4(log_dir, fig_dir)
    plot_fig5(log_dir, fig_dir)
    plot_fig6(log_dir, fig_dir)
    plot_fig7(log_dir, fig_dir)
    plot_fig8(log_dir, fig_dir)
    plot_fig9(log_dir, fig_dir)
    print("Done.\n")


if __name__ == "__main__":
    generate_all_figures()

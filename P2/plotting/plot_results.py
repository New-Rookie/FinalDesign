"""
Plot generation for Research Content 2 experiments.

Produces 5 figures:
  Fig1 — RF estimator effectiveness (predicted vs measured PRR)
  Fig2 — GMAPPO reward curve under learning-rate sweep
  Fig3 — LA_pi vs N_total (algorithm comparison)
  Fig4 — LA_pi vs eta_ch (algorithm comparison)
  Fig5 — GMAPPO convergence under weight ratio sweep (w_Q : w_S)
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

ALGO_MARKERS = {"GMAPPO": "o", "MAPPO": "s", "Greedy": "^",
                "ACO": "D", "GA": "v"}
ALGO_COLORS = {"GMAPPO": "#E91E63", "MAPPO": "#2196F3",
               "Greedy": "#FF9800", "ACO": "#4CAF50", "GA": "#9C27B0"}

LR_COLORS = {1e-4: "#2196F3", 3e-4: "#FF9800", 1e-3: "#4CAF50"}
LR_LABELS = {1e-4: "lr=1e-4", 3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}

LC_COLORS = {"satellite": "#E91E63", "uav_terrestrial": "#2196F3",
             "sea_surface": "#4CAF50", "terrestrial": "#FF9800"}


def _safe_load(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — RF estimator effectiveness
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_p_raw.csv"))
    df_s = _safe_load(os.path.join(log_dir, "block_p_summary.csv"))
    if df is None:
        print("  [Fig1] block_p_raw.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Ideal")

    for lc in ["satellite", "uav_terrestrial", "sea_surface", "terrestrial"]:
        sub = df[df["link_class"] == lc]
        if sub.empty:
            continue
        ax.scatter(sub["prr_true"], sub["prr_pred"],
                   s=4, alpha=0.3, color=LC_COLORS.get(lc, "gray"),
                   label=lc.replace("_", " "))

    caption_parts = []
    if df_s is not None:
        for _, row in df_s.iterrows():
            caption_parts.append(
                f"{row['link_class']}: R²={row['R2']:.3f}, "
                f"MAE={row['MAE']:.3f}")

    ax.set_xlabel("Measured $PRR_{emp}$")
    ax.set_ylabel("Predicted $\\hat{PRR}$")
    ax.set_title("RF Link-Quality Estimator Effectiveness")
    ax.legend(loc="lower right", markerscale=3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if caption_parts:
        ax.text(0.02, 0.98, "\n".join(caption_parts),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

    fig.savefig(os.path.join(fig_dir, "Fig1_RC2_RF_effectiveness.png"))
    plt.close(fig)
    print("  [Fig1] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Reward curve (LR sweep)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig2(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_a_summary.csv"))
    if df is None:
        print("  [Fig2] block_a_summary.csv not found, skipping.")
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
    ax.set_title("GMAPPO — Reward Curve (Learning Rate Sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig2_RC2_reward_lr.png"))
    plt.close(fig)
    print("  [Fig2] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — LA vs N_total (algorithm comparison)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig3(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_b_summary.csv"))
    if df is None:
        print("  [Fig3] block_b_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["GMAPPO", "MAPPO", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("N_total")
        if sub.empty:
            continue
        ax.errorbar(sub["N_total"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=algo, capsize=3, linewidth=1.5)

    ax.set_xlabel("Total node count $N_{total}$")
    ax.set_ylabel("Mean Link Advantage $LA_{\\pi}$")
    ax.set_title("Algorithm Comparison — LA vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig3_RC2_LA_vs_nodes.png"))
    plt.close(fig)
    print("  [Fig3] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — LA vs eta_ch (algorithm comparison)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig4(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_c_summary.csv"))
    if df is None:
        print("  [Fig4] block_c_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["GMAPPO", "MAPPO", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("eta_ch")
        if sub.empty:
            continue
        ax.errorbar(sub["eta_ch"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS.get(algo, "o"),
                    color=ALGO_COLORS.get(algo, "gray"),
                    label=algo, capsize=3, linewidth=1.5)

    ax.set_xlabel("Channel-condition scale $\\eta_{ch}$")
    ax.set_ylabel("Mean Link Advantage $LA_{\\pi}$")
    ax.set_title("Algorithm Comparison — LA vs Channel Condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig4_RC2_LA_vs_channel.png"))
    plt.close(fig)
    print("  [Fig4] saved.")




# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — GMAPPO convergence under weight ratio sweep
# ═══════════════════════════════════════════════════════════════════════════

WQ_COLORS = {0.2: "#E91E63", 0.3: "#FF9800", 0.5: "#4CAF50",
             0.7: "#2196F3", 0.8: "#9C27B0"}

def plot_fig5(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_d_summary.csv"))
    if df is None:
        print("  [Fig5] block_d_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for w_q in sorted(df["w_Q"].unique()):
        w_s = round(1.0 - w_q, 1)
        sub = df[df["w_Q"] == w_q].sort_values("episode")
        if sub.empty:
            continue
        label = f"$w_Q$:{w_q} / $w_S$:{w_s}"
        color = WQ_COLORS.get(w_q, "gray")
        ax.plot(sub["episode"], sub["mean_reward"],
                color=color, label=label, linewidth=1.5)
        if "std_reward" in sub.columns:
            ax.fill_between(sub["episode"],
                            sub["mean_reward"] - sub["std_reward"],
                            sub["mean_reward"] + sub["std_reward"],
                            alpha=0.12, color=color)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("GMAPPO Convergence — Weight Ratio Sweep ($w_Q : w_S$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig5_RC2_wQ_wS_convergence.png"))
    plt.close(fig)
    print("  [Fig5] saved.")

# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(log_dir: str = "P2/logs",
                         fig_dir: str = "P2/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating Research Content 2 figures...")
    plot_fig1(log_dir, fig_dir)
    plot_fig2(log_dir, fig_dir)
    plot_fig3(log_dir, fig_dir)
    plot_fig4(log_dir, fig_dir)
    plot_fig5(log_dir, fig_dir)
    print("Done.\n")


if __name__ == "__main__":
    generate_all_figures()

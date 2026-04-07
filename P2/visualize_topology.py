"""
Standalone network topology visualizer for P2 (Link Selection).

Generates a publication-quality network topology figure showing:
  - All nodes colored/shaped by type (satellite, UAV, ship, buoy, land)
  - Ground-truth topology edges (SNR >= gamma_link & contact duration >= T_min)
  - Candidate service paths for a selected source buoy (optional)

Usage (from V11/ directory):
    python -m P2.visualize_topology
    python -m P2.visualize_topology --n-total 30 --seed 42 --show-paths
    python -m P2.visualize_topology --save topo.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P2.link_quality.path_manager import PathManager


# ── visual config ────────────────────────────────────────────────────────

TYPE_STYLE = {
    "satellite": {"color": "#9467bd", "marker": "^", "size": 120, "label": "Satellite (LEO)"},
    "uav":       {"color": "#ff7f0e", "marker": "D", "size": 90,  "label": "UAV"},
    "ship":      {"color": "#1f77b4", "marker": "s", "size": 90,  "label": "Ship"},
    "buoy":      {"color": "#2ca02c", "marker": "o", "size": 60,  "label": "Buoy"},
    "land":      {"color": "#d62728", "marker": "P", "size": 110, "label": "Land Station"},
}

PATH_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def build_topology(cfg: EnvConfig, seed: int = 0):
    """Create env, reset, return (env, nodes, gt_adj, link_phy)."""
    env = MarineIoTEnv(cfg, mode="link_selection", max_steps=200)
    env.reset(seed=seed)
    nodes = env.nodes
    gt_adj = env.get_ground_truth_topology()
    link_phy = env.link_phy
    return env, nodes, gt_adj, link_phy


def draw_topology(nodes, gt_adj, service_paths=None, source_id=None,
                  title=None, save_path=None, show=True):
    """Draw the network topology with matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(10, 9))

    n = len(nodes)
    pos_2d = {nd.node_id: (nd.position[0], nd.position[1]) for nd in nodes}
    type_map = {nd.node_id: nd.node_type for nd in nodes}

    # -- edges (GT topology) --
    for i in range(n):
        for j in range(i + 1, n):
            if gt_adj[i, j]:
                ni, nj = nodes[i].node_id, nodes[j].node_id
                xi, yi = pos_2d[ni]
                xj, yj = pos_2d[nj]
                ax.plot([xi, xj], [yi, yj], color="#cccccc", linewidth=0.6,
                        zorder=1, alpha=0.7)

    # -- service paths overlay --
    if service_paths:
        for idx, sp in enumerate(service_paths):
            color = PATH_COLORS[idx % len(PATH_COLORS)]
            hops = sp.hops
            for (a, b) in hops:
                if a in pos_2d and b in pos_2d:
                    xa, ya = pos_2d[a]
                    xb, yb = pos_2d[b]
                    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                                arrowprops=dict(arrowstyle="->", color=color,
                                                lw=1.8, shrinkA=4, shrinkB=4),
                                zorder=3)

    # -- nodes --
    legend_handles = []
    drawn_types = set()
    for nd in nodes:
        style = TYPE_STYLE.get(nd.node_type, {"color": "gray", "marker": "o", "size": 50})
        x, y = pos_2d[nd.node_id]

        highlight = (source_id is not None and nd.node_id == source_id)
        edge_color = "red" if highlight else "white"
        lw = 2.0 if highlight else 0.5

        ax.scatter(x, y, c=style["color"], marker=style["marker"],
                   s=style["size"] * (1.8 if highlight else 1.0),
                   edgecolors=edge_color, linewidths=lw, zorder=5)

        ax.annotate(str(nd.node_id), (x, y), fontsize=6, ha="center",
                    va="bottom", xytext=(0, 4), textcoords="offset points",
                    color="#333333", zorder=6)

        if nd.node_type not in drawn_types:
            drawn_types.add(nd.node_type)
            legend_handles.append(
                mlines.Line2D([], [], color=style["color"],
                              marker=style["marker"], linestyle="None",
                              markersize=8, label=style["label"]))

    legend_handles.sort(key=lambda h: h.get_label())

    if service_paths:
        legend_handles.append(
            mlines.Line2D([], [], color=PATH_COLORS[0], linewidth=1.8,
                          label=f"Service paths (buoy {source_id})"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")

    edge_count = int(np.triu(gt_adj, k=1).sum())
    if title is None:
        title = f"P2 Network Topology  (N={len(nodes)}, edges={edge_count})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Topology figure saved to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="P2 Network Topology Visualizer")
    parser.add_argument("--n-total", type=int, default=20,
                        help="Total node count (default: 20, same as P2 experiments)")
    parser.add_argument("--eta-ch", type=float, default=1.0,
                        help="Channel condition scale")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for node placement")
    parser.add_argument("--show-paths", action="store_true",
                        help="Overlay candidate service paths for one source buoy")
    parser.add_argument("--buoy-id", type=int, default=None,
                        help="Specific buoy node ID to show paths for (auto-pick if omitted)")
    parser.add_argument("--max-paths", type=int, default=6,
                        help="Max service paths to draw (default: 6)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file (e.g. topo.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open the plot window")
    args = parser.parse_args()

    cfg = EnvConfig(N_total=args.n_total, eta_ch=args.eta_ch,
                    print_diagnostics=False)
    env, nodes, gt_adj, link_phy = build_topology(cfg, seed=args.seed)

    print(f"Nodes: {len(nodes)}")
    counts = cfg.node_counts
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:12s}: {cnt}")
    edge_count = int(np.triu(gt_adj, k=1).sum())
    print(f"GT edges: {edge_count}")

    service_paths = None
    source_id = None

    if args.show_paths:
        path_mgr = PathManager(cfg)
        buoy_ids = [nd.node_id for nd in nodes if nd.node_type == "buoy"]

        if args.buoy_id is not None and args.buoy_id in buoy_ids:
            source_id = args.buoy_id
        elif buoy_ids:
            source_id = buoy_ids[0]

        if source_id is not None:
            all_paths = path_mgr.enumerate_paths(nodes, link_phy, [source_id])
            paths = all_paths.get(source_id, [])
            service_paths = paths[:args.max_paths]
            print(f"\nSource buoy: {source_id}")
            print(f"Feasible paths: {len(paths)}  (showing {len(service_paths)})")
            for i, sp in enumerate(service_paths):
                seq = " -> ".join(str(nid) for nid in sp.node_sequence)
                print(f"  Path {i}: {seq}  ({sp.hop_count} hops)")

    draw_topology(nodes, gt_adj, service_paths=service_paths,
                  source_id=source_id, save_path=args.save,
                  show=not args.no_show)

    env.close()


if __name__ == "__main__":
    main()

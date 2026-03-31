"""
visualize.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Step 6: Generate publication-ready figures.

Figures produced
----------------
fig1_schema_dags.png
    2x3 grid of all 6 tradition schemas drawn as directed acyclic graphs.
    Nodes coloured by functional_role; edges coloured by relationship type.
    Hierarchical layout (root at top, matter at bottom).

fig2_ged_heatmap.png
    6x6 heatmap of normalised structural Graph Edit Distance.
    Lower = more similar. Annotated with raw GED values.

fig3_similarity_triptych.png
    Three side-by-side heatmaps: structural GED | WL-role similarity |
    role-sequence Levenshtein distance.

fig4_depth_violin.png
    Real schema depths as labelled points overlaid on the pooled null
    distribution violin. Visualises Test 4 (Mann-Whitney U) result.

fig5_dendrogram.png
    Hierarchical clustering dendrogram built from the GED distance matrix.
    Cuts into 2 topology families (linear chain / branching tree).

fig6_subgraph_nesting.png
    Directed graph of confirmed subgraph-isomorphism relations between
    traditions. Edges mean "A is a structural subgraph of B".

Outputs
-------
  outputs/figures/fig*.png  (300 dpi, RGB)

Usage:
    python visualize.py
"""

import json
import os
import sys

# Headless backend -- MUST be set before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from encode_schemas import load_all_schemas, SCHEMAS_DIR
from compute_invariants import compute_invariants

FIGURES_DIR = os.path.join(_HERE, "..", "outputs", "figures")
SIM_DIR = os.path.join(_HERE, "..", "outputs", "similarity_matrix")
CONTROLS_DIR = os.path.join(_HERE, "..", "outputs", "invariants", "controls")

DPI = 300


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

ROLE_COLORS = {
    "source":         "#3B1A6E",   # deep violet
    "first_emanation":"#5C4AE4",   # blue-violet
    "intellect":      "#1565C0",   # deep blue
    "soul":           "#00838F",   # teal
    "intermediary":   "#2E7D32",   # forest green
    "fallen":         "#E65100",   # deep orange (crisis)
    "demiurge":       "#B71C1C",   # deep red
    "matter":         "#37474F",   # dark slate
    "process":        "#F57F17",   # amber (transformation)
}

EDGE_COLORS = {
    "emanation":      "#78909C",   # blue-grey
    "creation":       "#8D6E63",   # warm brown
    "fragmentation":  "#D32F2F",   # red (rupture)
    "contraction":    "#7B1FA2",   # purple (Tzimtzum)
    "reflection":     "#0288D1",   # sky blue
    "succession":     "#FF8F00",   # amber (theogonic displacement, WP 1.3)
}

DEFAULT_ROLE_COLOR = "#9E9E9E"
DEFAULT_EDGE_COLOR = "#BDBDBD"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def hierarchical_layout(G: nx.DiGraph, x_spacing: float = 2.2,
                         y_spacing: float = 1.8) -> dict:
    """
    Manual Sugiyama-style top-down layout.
    Root at y=0; each level descends by y_spacing.
    Nodes at the same level are centred and spaced by x_spacing.
    Returns {node: (x, y)}.
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        return {n: (i, 0) for i, n in enumerate(G.nodes)}
    root = roots[0]

    levels = nx.single_source_shortest_path_length(G, root)
    by_level: dict[int, list] = {}
    for node, lv in levels.items():
        by_level.setdefault(lv, []).append(node)

    pos = {}
    for lv, nodes in by_level.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes, key=str)):
            x = (i - (n - 1) / 2.0) * x_spacing
            y = -lv * y_spacing
            pos[node] = (x, y)
    return pos


# ---------------------------------------------------------------------------
# Figure 1: Schema DAG grid
# ---------------------------------------------------------------------------

def fig1_schema_dags(schemas: dict) -> str:
    """Grid of all tradition DAGs (adapts to corpus size)."""
    tradition_order = [
        "plotinian",
        "proclan_neoplatonic",
        "chaldean",
        "hermetic",
        "lurianic_kabbalistic",
        "sethian_gnostic",
        "valentinian_gnostic",
        "samkhya",
        "taoist_ddj",
        "ishraq_illuminationist",
        "genesis_creationist",
        "bundahishn_zoroastrian",
        "manichaean",
        "derveni_orphic",
        # WP 1.4 additions
        "popol_vuh_maya",
        "trimorphic_protennoia",
        "gospel_of_mary",
    ]
    traditions = [t for t in tradition_order if t in schemas]
    # Append any not in the order list
    traditions += [t for t in sorted(schemas) if t not in traditions]

    n_schemas = len(traditions)
    n_cols = 4 if n_schemas > 9 else 3
    n_rows = (n_schemas + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows + 2))
    axes = axes.flatten()

    for idx, tradition in enumerate(traditions):
        ax = axes[idx]
        G = schemas[tradition]
        pos = hierarchical_layout(G)

        node_colors = [
            ROLE_COLORS.get(G.nodes[n].get("functional_role", ""), DEFAULT_ROLE_COLOR)
            for n in G.nodes
        ]

        # Draw edges first (behind nodes)
        for u, v, attrs in G.edges(data=True):
            rel = attrs.get("relationship", "")
            color = EDGE_COLORS.get(rel, DEFAULT_EDGE_COLOR)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                ax=ax, edge_color=color, width=2.0,
                arrows=True, arrowsize=18, arrowstyle="-|>",
                connectionstyle="arc3,rad=0.05",
                node_size=900,
            )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors, node_size=900,
            edgecolors="#FFFFFF", linewidths=1.5,
        )

        # Labels: use node id (already short)
        labels = {n: n.replace("_", "\n") for n in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=5.5, font_color="white", font_weight="bold",
        )

        # Short tradition title
        short_title = tradition.replace("_", " ").title()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        depth = max(nx.single_source_shortest_path_length(
            G, [n for n in G.nodes if G.in_degree(n) == 0][0]).values())
        ax.set_title(
            f"{short_title}\n"
            f"N={n_nodes}  E={n_edges}  D={depth}",
            fontsize=8, fontweight="bold", pad=6,
        )
        ax.axis("off")
        ax.set_facecolor("#F8F8F8")

    # Hide any unused axes
    for idx in range(len(traditions), len(axes)):
        axes[idx].set_visible(False)

    # Role legend (bottom of figure)
    role_patches = [
        mpatches.Patch(color=color, label=role.replace("_", " "))
        for role, color in ROLE_COLORS.items()
    ]
    edge_patches = [
        mpatches.Patch(color=color, label=rel)
        for rel, color in EDGE_COLORS.items()
    ]
    fig.legend(
        handles=role_patches + edge_patches,
        loc="lower center", ncol=7,
        fontsize=7, title="Node role  |  Edge type",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, 0.01),
        framealpha=0.9,
    )

    fig.suptitle(
        "Emanation-Schema DAGs: Six Cosmological Traditions",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    path = os.path.join(FIGURES_DIR, "fig1_schema_dags.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: GED heatmap
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ged_matrix(traditions: list) -> np.ndarray:
    ged_data = _load_json(os.path.join(SIM_DIR, "structural_ged.json"))
    n = len(traditions)
    M = np.zeros((n, n))
    for i, a in enumerate(traditions):
        for j, b in enumerate(traditions):
            if i == j:
                M[i, j] = 0
            else:
                key = f"{a} vs {b}" if a < b else f"{b} vs {a}"
                M[i, j] = ged_data.get(key, {}).get("structural_ged", 0)
    return M


def _norm_ged_matrix(traditions: list) -> np.ndarray:
    ged_data = _load_json(os.path.join(SIM_DIR, "structural_ged.json"))
    n = len(traditions)
    M = np.zeros((n, n))
    for i, a in enumerate(traditions):
        for j, b in enumerate(traditions):
            if i == j:
                M[i, j] = 0
            else:
                key = f"{a} vs {b}" if a < b else f"{b} vs {a}"
                M[i, j] = ged_data.get(key, {}).get("normalized_structural_ged", 0)
    return M


def fig2_ged_heatmap(traditions: list) -> str:
    """6x6 normalised GED heatmap."""
    M_norm = _norm_ged_matrix(traditions)
    M_raw = _ged_matrix(traditions)

    # Custom green-white-red colormap (green=similar, red=distant)
    cmap = LinearSegmentedColormap.from_list(
        "ged", ["#1B5E20", "#A5D6A7", "#FFFFFF", "#EF9A9A", "#B71C1C"]
    )

    short = [t.replace("_", "\n") for t in traditions]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M_norm, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")

    # Annotate cells with raw GED
    for i in range(len(traditions)):
        for j in range(len(traditions)):
            val = M_raw[i, j]
            text_color = "white" if M_norm[i, j] > 0.5 or M_norm[i, j] == 0 else "black"
            ax.text(j, i, f"{int(val)}" if val == int(val) else f"{val:.1f}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=text_color)

    ax.set_xticks(range(len(traditions)))
    ax.set_yticks(range(len(traditions)))
    ax.set_xticklabels(short, fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title("Structural Graph Edit Distance (raw values annotated)\n"
                 "Colour = normalised GED: green = more similar, red = more distant",
                 fontsize=9, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised GED", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_ged_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Similarity triptych
# ---------------------------------------------------------------------------

def fig3_similarity_triptych(traditions: list) -> str:
    """Three-panel figure: GED | WL-roles | Role-Levenshtein."""
    wl_data = _load_json(os.path.join(SIM_DIR, "wl_roles.json"))
    lev_data = _load_json(os.path.join(SIM_DIR, "role_levenshtein.json"))
    n = len(traditions)

    # Normalised GED (similarity = 1 - nGED)
    M_ged_sim = 1.0 - _norm_ged_matrix(traditions)
    np.fill_diagonal(M_ged_sim, 1.0)

    # WL role similarity
    M_wl = np.zeros((n, n))
    for i, a in enumerate(traditions):
        for j, b in enumerate(traditions):
            if i == j:
                M_wl[i, j] = 1.0
            else:
                key = f"{a} vs {b}" if a < b else f"{b} vs {a}"
                M_wl[i, j] = wl_data.get(key, {}).get("wl_role_similarity", 0)

    # Role-sequence similarity (1 - norm_lev)
    M_lev = np.zeros((n, n))
    for i, a in enumerate(traditions):
        for j, b in enumerate(traditions):
            if i == j:
                M_lev[i, j] = 1.0
            else:
                key = f"{a} vs {b}" if a < b else f"{b} vs {a}"
                M_lev[i, j] = 1.0 - lev_data.get(key, {}).get(
                    "normalized_role_seq_distance", 0)

    short = [t.replace("_neoplatonic", "").replace("_gnostic", "")
               .replace("_kabbalistic", "").replace("_", "\n")
             for t in traditions]

    cmap_sim = LinearSegmentedColormap.from_list(
        "sim", ["#FFFFFF", "#A5D6A7", "#1B5E20"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    titles = [
        "GED Similarity\n(1 - normalised GED)",
        "WL Role Similarity\n(3-iteration kernel)",
        "Role Sequence Similarity\n(1 - normalised Levenshtein)",
    ]
    matrices = [M_ged_sim, M_wl, M_lev]

    for ax, title, M in zip(axes, titles, matrices):
        im = ax.imshow(M, cmap=cmap_sim, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                color = "white" if M[i, j] > 0.65 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}",
                        ha="center", va="center", fontsize=7.5, color=color)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-Tradition Similarity: Three Complementary Metrics",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_similarity_triptych.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Depth violin plot
# ---------------------------------------------------------------------------

def fig4_depth_violin(schemas: dict) -> str:
    """Real schema depths vs. pooled null distribution."""
    pooled = _load_json(os.path.join(CONTROLS_DIR, "pooled_controls.json"))
    null_depths = [c["depth"] for c in pooled]

    real_depths = {}
    for t, G in schemas.items():
        root = [n for n in G.nodes if G.in_degree(n) == 0][0]
        d = max(nx.single_source_shortest_path_length(G, root).values())
        real_depths[t] = d

    fig, ax = plt.subplots(figsize=(9, 6))

    # Violin for null distribution
    parts = ax.violinplot([null_depths], positions=[0], widths=0.6,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#B0BEC5")
        pc.set_alpha(0.6)
        pc.set_edgecolor("#546E7A")
    parts["cmedians"].set_color("#37474F")
    parts["cmedians"].set_linewidth(2)
    parts["cmaxes"].set_color("#78909C")
    parts["cmins"].set_color("#78909C")
    parts["cbars"].set_color("#78909C")

    # Real schemas as coloured points with jitter
    jitter_x = np.linspace(-0.15, 0.15, len(real_depths))
    for jx, (t, d) in zip(jitter_x, sorted(real_depths.items(), key=lambda x: x[1])):
        short_t = t.replace("_neoplatonic", "").replace("_gnostic", "") \
                   .replace("_kabbalistic", "").replace("_", " ")
        ax.scatter(jx, d, s=140, zorder=5, color="#1B5E20", edgecolors="white",
                   linewidths=1.5)
        ax.annotate(short_t, (jx, d), textcoords="offset points",
                    xytext=(12, 2), fontsize=7.5, color="#1B5E20",
                    fontweight="bold")

    ax.set_xticks([0])
    ax.set_xticklabels(["Pooled null\n(n=1,000 random trees)"], fontsize=9)
    ax.set_ylabel("Graph Depth (longest path root → leaf, edges)", fontsize=9)
    ax.set_title("Real Schema Depths vs. Pooled Null Distribution\n"
                 "Green points = actual traditions; grey violin = random trees "
                 "(Mann-Whitney U, p=0.012)",
                 fontsize=9, fontweight="bold")

    # Null mean line
    null_mean = np.mean(null_depths)
    ax.axhline(null_mean, color="#78909C", linestyle="--", linewidth=1,
               label=f"Null mean = {null_mean:.2f}")
    ax.legend(fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, 0.9)
    ax.set_ylim(0, max(real_depths.values()) + 1.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_depth_violin.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: Dendrogram
# ---------------------------------------------------------------------------

def fig5_dendrogram(traditions: list) -> str:
    """Hierarchical clustering from GED distance matrix."""
    M = _ged_matrix(traditions)

    # scipy linkage needs condensed distance matrix
    condensed = squareform(M)
    Z = sch.linkage(condensed, method="average")

    short = [
        t.replace("_neoplatonic", "").replace("_gnostic", "")
         .replace("_kabbalistic", "").replace("_", " ").title()
        for t in traditions
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    sch.dendrogram(
        Z, labels=short, ax=ax,
        color_threshold=4.5,   # cuts into the 2 topology families
        above_threshold_color="#9E9E9E",
        leaf_font_size=9,
        leaf_rotation=0,
    )

    ax.axhline(4.5, color="#D32F2F", linestyle="--", linewidth=1.2,
               label="Family boundary (GED = 4.5)")
    ax.set_ylabel("Structural GED (average linkage)", fontsize=9)
    ax.set_title("Topology Family Clustering (UPGMA on Structural GED)\n"
                 "Red dashed line separates linear-chain family from branching-tree family",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_dendrogram.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: Subgraph nesting diagram
# ---------------------------------------------------------------------------

def fig6_subgraph_nesting(traditions: list) -> str:
    """Directed graph of confirmed subgraph-isomorphism nesting relations."""
    sub_data = _load_json(os.path.join(SIM_DIR, "subgraph_membership.json"))
    confirmed = sub_data.get("confirmed_subgraph_relations", {})

    # Build nesting graph
    H = nx.DiGraph()
    H.add_nodes_from(traditions)
    for rel in confirmed:
        # "A in B" means A is a subgraph of B
        parts = rel.split(" in ")
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a in traditions and b in traditions:
                H.add_edge(a, b)  # A ⊂ B

    # Remove transitive edges for cleaner diagram (transitive reduction)
    try:
        H_reduced = nx.transitive_reduction(H)
    except Exception:
        H_reduced = H

    # Layout by node count (number of nodes in each schema)
    schema_sizes = {t: G.number_of_nodes() for t, G in _SCHEMAS.items() if t in traditions}
    pos = {}
    by_size: dict = {}
    for t, sz in schema_sizes.items():
        by_size.setdefault(sz, []).append(t)

    for sz, nodes in by_size.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2.0) * 3.5
            y = sz * 1.5
            pos[node] = (x, y)

    short = {
        t: t.replace("_neoplatonic", "").replace("_gnostic", "")
             .replace("_kabbalistic", "").replace("_", "\n").title()
        for t in traditions
    }
    node_colors = [ROLE_COLORS.get("source", "#3B1A6E")] * len(traditions)
    node_colors = ["#5C4AE4" if schema_sizes.get(t, 0) <= 6 else
                   "#1565C0" if schema_sizes.get(t, 0) <= 7 else
                   "#00838F" if schema_sizes.get(t, 0) <= 8 else
                   "#2E7D32"
                   for t in traditions]

    fig, ax = plt.subplots(figsize=(10, 7))

    nx.draw_networkx_nodes(
        H_reduced, pos, ax=ax,
        node_color=node_colors, node_size=2200,
        edgecolors="white", linewidths=2,
        nodelist=list(traditions),
    )
    nx.draw_networkx_labels(
        H_reduced, pos, labels=short, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )
    nx.draw_networkx_edges(
        H_reduced, pos, ax=ax,
        edge_color="#D32F2F", width=2.0,
        arrows=True, arrowsize=20, arrowstyle="-|>",
        node_size=2200,
        connectionstyle="arc3,rad=0.1",
    )

    # Y-axis labels
    for sz in sorted(set(schema_sizes.values())):
        ax.text(-5.8, sz * 1.5, f"N={sz}", va="center", fontsize=8,
                color="#37474F", fontstyle="italic")

    ax.set_title("Subgraph Nesting Relations (Transitive Reduction)\n"
                 "A → B means A is structurally contained within B\n"
                 "Y-axis = number of nodes (N) in each schema",
                 fontsize=9, fontweight="bold")
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_subgraph_nesting.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

_SCHEMAS: dict = {}   # module-level cache so fig6 can access node counts


def run_all(schemas_dir: str = SCHEMAS_DIR) -> list[str]:
    global _SCHEMAS

    os.makedirs(FIGURES_DIR, exist_ok=True)
    raw_schemas = load_all_schemas(schemas_dir)
    graphs = {t: G for t, (G, _) in raw_schemas.items()}
    _SCHEMAS = graphs

    traditions = sorted(graphs.keys())
    paths = []

    print("  Generating fig1: schema DAGs...")
    paths.append(fig1_schema_dags(graphs))

    print("  Generating fig2: GED heatmap...")
    paths.append(fig2_ged_heatmap(traditions))

    print("  Generating fig3: similarity triptych...")
    paths.append(fig3_similarity_triptych(traditions))

    print("  Generating fig4: depth violin...")
    paths.append(fig4_depth_violin(graphs))

    print("  Generating fig5: dendrogram...")
    paths.append(fig5_dendrogram(traditions))

    print("  Generating fig6: subgraph nesting...")
    paths.append(fig6_subgraph_nesting(traditions))

    return paths


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    print("=" * 60)
    print("  WP 1.1 -- visualize.py")
    print("  Publication-Ready Figure Generator")
    print("=" * 60)
    print()

    import time
    t0 = time.time()
    paths = run_all()
    elapsed = time.time() - t0

    print()
    print(f"  Generated {len(paths)} figures in {elapsed:.2f}s")
    print()
    for p in paths:
        fname = os.path.basename(p)
        size_kb = os.path.getsize(p) // 1024
        print(f"  {fname:<35}  {size_kb:>5} KB")
    print()
    print(f"  Output directory: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

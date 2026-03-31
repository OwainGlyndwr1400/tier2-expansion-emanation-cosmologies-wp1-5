"""
generate_controls.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Step 3: Generate randomized DAG null models for statistical comparison.

Null model
----------
All 6 real schemas are rooted DAG trees (edges = nodes - 1).
The null model is therefore **uniformly random labeled rooted trees**,
generated via nx.random_labeled_rooted_tree() (Prufer sequence method).
Edges are oriented away from the root to produce a valid DAG.

Two control sets
----------------
1. Pooled controls (N=1000):
   Node count sampled from the empirical corpus distribution:
   {5, 6, 7, 8, 8, 9} -- drawn with replacement (8 has double weight).
   Tests whether the corpus as a family clusters unusually.

2. Stratified controls (N=1000 per tradition, 6000 total):
   Node count fixed to match each tradition.
   Tests whether each tradition's topology is unusual for its size class.

Invariants computed per control DAG
------------------------------------
  node_count, edge_count, depth, diameter, width,
  max_branching, mean_branching, leaf_count, branching_node_count,
  is_linear_chain, in_degree_sequence, out_degree_sequence,
  depth_profile (level -> node_count)

Outputs
-------
  outputs/invariants/controls/pooled_controls.json
  outputs/invariants/controls/stratified_{tradition}.json    (x6)
  outputs/invariants/controls/control_stats.json             (summary)

Usage:
    python generate_controls.py
"""

import json
import os
import random
import sys
import time

import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from encode_schemas import load_all_schemas, SCHEMAS_DIR

CONTROLS_DIR = os.path.join(_HERE, "..", "outputs", "invariants", "controls")

N_POOLED = 1000
N_STRATIFIED = 1000

# Empirical node-count distribution of the corpus (22 schemas, WP 1.5)
# WP 1.4 (17): [5,5,5,6,6,7,7,7,7,8,8,8,9,9,9,11,11]
# WP 1.5 adds: Nasadiya (5), Purusha (7), Revelation (6), Mi'raj (10), 1 Enoch (6)
CORPUS_NODE_COUNTS = [5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 11, 11]


# ---------------------------------------------------------------------------
# Random DAG tree generator
# ---------------------------------------------------------------------------

def random_dag_tree(n: int, seed=None) -> nx.DiGraph:
    """
    Generate a uniformly random labeled rooted DAG tree on n nodes.

    Uses nx.random_labeled_rooted_tree() (Prufer-sequence based) for
    uniform sampling over all labeled trees, then orients edges away
    from the designated root via BFS.

    Returns a nx.DiGraph with a single root (in-degree 0) and all
    edges directed top-down.
    """
    if n == 1:
        G = nx.DiGraph()
        G.add_node(0)
        return G

    T = nx.random_labeled_rooted_tree(n, seed=seed)
    root = T.graph["root"]

    # Orient edges away from root -- bfs_tree guarantees a valid DAG
    dag = nx.bfs_tree(T, root)
    return dag


# ---------------------------------------------------------------------------
# Invariant computation for control DAGs (structural metrics only -- no roles)
# ---------------------------------------------------------------------------

def _dag_depth_profile(G: nx.DiGraph) -> dict[int, int]:
    """Return {level: node_count} using shortest path from root."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        return {}
    lengths = nx.single_source_shortest_path_length(G, roots[0])
    profile: dict[int, int] = {}
    for _node, level in lengths.items():
        profile[level] = profile.get(level, 0) + 1
    return dict(sorted(profile.items()))


def control_invariants(G: nx.DiGraph) -> dict:
    """
    Compute structural invariants for a single control DAG.
    Keeps only the metrics needed for statistical comparison.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    depth = nx.dag_longest_path_length(G)
    profile = _dag_depth_profile(G)
    width = max(profile.values()) if profile else 0
    leaves = [node for node in G.nodes if G.out_degree(node) == 0]
    branching = [node for node in G.nodes if G.out_degree(node) > 1]

    non_leaf_out = [G.out_degree(node) for node in G.nodes if G.out_degree(node) > 0]
    mean_br = round(sum(non_leaf_out) / len(non_leaf_out), 4) if non_leaf_out else 0.0

    try:
        diameter = nx.diameter(G.to_undirected())
    except nx.NetworkXError:
        diameter = depth

    return {
        "node_count": n,
        "edge_count": G.number_of_edges(),
        "depth": depth,
        "diameter": diameter,
        "width": width,
        "max_branching": max(G.out_degree(node) for node in G.nodes),
        "mean_branching": mean_br,
        "leaf_count": len(leaves),
        "branching_node_count": len(branching),
        "is_linear_chain": len(branching) == 0,
        "in_degree_sequence": sorted(d for _, d in G.in_degree()),
        "out_degree_sequence": sorted(d for _, d in G.out_degree()),
        "depth_profile": {str(k): v for k, v in profile.items()},
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_pooled(n_controls: int = N_POOLED, seed: int = 42) -> list[dict]:
    """
    Generate n_controls random DAG trees with node counts sampled
    from the empirical corpus distribution.
    """
    rng = random.Random(seed)
    controls = []
    for i in range(n_controls):
        n = rng.choice(CORPUS_NODE_COUNTS)
        tree_seed = rng.randint(0, 2**31)
        G = random_dag_tree(n, seed=tree_seed)
        inv = control_invariants(G)
        inv["control_id"] = f"pooled_{i:04d}"
        controls.append(inv)
    return controls


def generate_stratified(n: int, n_controls: int = N_STRATIFIED, seed: int = 0) -> list[dict]:
    """
    Generate n_controls random DAG trees, all with exactly n nodes.
    """
    rng = random.Random(seed + n * 997)   # tradition-specific seed offset
    controls = []
    for i in range(n_controls):
        tree_seed = rng.randint(0, 2**31)
        G = random_dag_tree(n, seed=tree_seed)
        inv = control_invariants(G)
        inv["control_id"] = f"n{n}_{i:04d}"
        controls.append(inv)
    return controls


# ---------------------------------------------------------------------------
# Summary statistics over a control set
# ---------------------------------------------------------------------------

METRICS = [
    "depth", "diameter", "width",
    "max_branching", "mean_branching",
    "leaf_count", "branching_node_count",
]


def _percentile(values: list[float], p: float) -> float:
    """Simple percentile without numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


def summary_stats(controls: list[dict], label: str = "") -> dict:
    """
    Compute mean, std, and percentiles (5, 25, 50, 75, 95) for each metric.
    Also counts linear-chain frequency.
    """
    stats: dict = {"label": label, "n": len(controls)}
    for metric in METRICS:
        vals = [c[metric] for c in controls if metric in c]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = variance ** 0.5
        stats[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "p05": _percentile(vals, 5),
            "p25": _percentile(vals, 25),
            "p50": _percentile(vals, 50),
            "p75": _percentile(vals, 75),
            "p95": _percentile(vals, 95),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    chain_count = sum(1 for c in controls if c.get("is_linear_chain", False))
    stats["linear_chain_frequency"] = round(chain_count / len(controls), 4)
    return stats


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(schemas_dir: str = SCHEMAS_DIR, controls_dir: str = CONTROLS_DIR) -> dict:
    """
    Generate all control sets and save to disk.
    Returns summary statistics dict.
    """
    os.makedirs(controls_dir, exist_ok=True)

    # Load real schemas to get tradition node counts
    schemas = load_all_schemas(schemas_dir)
    tradition_node_counts = {
        t: data["validation"]["node_count"]
        for t, (_G, data) in schemas.items()
    }

    all_stats = {}

    # --- Pooled controls ---
    pooled = generate_pooled(N_POOLED, seed=42)
    _save(os.path.join(controls_dir, "pooled_controls.json"), pooled)
    all_stats["pooled"] = summary_stats(pooled, label="pooled")

    # --- Stratified controls per tradition ---
    for tradition, node_count in sorted(tradition_node_counts.items()):
        controls = generate_stratified(node_count, N_STRATIFIED, seed=42)
        fname = f"stratified_{tradition}.json"
        _save(os.path.join(controls_dir, fname), controls)
        all_stats[tradition] = summary_stats(
            controls, label=f"stratified_n{node_count}_{tradition}"
        )

    # --- Combined stats file ---
    stats_path = os.path.join(controls_dir, "control_stats.json")
    _save(stats_path, all_stats)

    return all_stats


def _save(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sep(char="-", width=72):
    print(char * width)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _sep("=")
    print("  WP 1.1 -- generate_controls.py")
    print("  Randomized DAG Null Model Generator")
    _sep("=")
    print(f"\n  Generating {N_POOLED} pooled controls + "
          f"{N_STRATIFIED} stratified per tradition (6 traditions)...")
    print(f"  Total control DAGs: {N_POOLED + N_STRATIFIED * 6:,}")
    print(f"  Node counts: pooled sampled from {CORPUS_NODE_COUNTS}")
    print()

    t0 = time.time()
    all_stats = run()
    elapsed = time.time() - t0

    print(f"  Generated in {elapsed:.2f}s")
    print()

    # --- Pooled summary ---
    print("  POOLED CONTROLS -- Summary Statistics (n=1000)")
    _sep()
    ps = all_stats["pooled"]
    _print_stats_row(ps)

    # --- Stratified summaries ---
    print()
    print("  STRATIFIED CONTROLS -- Summary Statistics (n=1000 per tradition)")
    _sep()
    print(f"  {'Tradition':<30} {'N nodes':>7} {'depth mu/sd':>12} {'width mu/sd':>12} "
          f"{'chain%':>7}")
    _sep()

    for key, stats in sorted(all_stats.items()):
        if key == "pooled":
            continue
        n_nodes = stats["label"].split("_n")[1].split("_")[0] if "_n" in stats["label"] else "?"
        tradition = key
        d = stats.get("depth", {})
        w = stats.get("width", {})
        chain_pct = round(stats.get("linear_chain_frequency", 0) * 100, 1)
        print(
            f"  {tradition:<30} {n_nodes:>7} "
            f"  {d.get('mean', 0):.2f}/{d.get('std', 0):.2f}   "
            f"  {w.get('mean', 0):.2f}/{w.get('std', 0):.2f}  "
            f"{chain_pct:>6}%"
        )

    _sep()

    # --- Real schema values vs. pooled null ---
    print()
    print("  REAL SCHEMA VALUES vs. POOLED NULL DISTRIBUTION")
    _sep()

    schemas = load_all_schemas()
    from compute_invariants import compute_invariants as _ci
    print(f"  {'Tradition':<30} {'depth':>6} {'p(<=)':>6}  {'width':>6} {'p(<=)':>6}  "
          f"{'chain':>6}")
    _sep()

    pooled_depths = [c["depth"] for c in _load(
        os.path.join(CONTROLS_DIR, "pooled_controls.json")
    )]
    pooled_widths = [c["width"] for c in _load(
        os.path.join(CONTROLS_DIR, "pooled_controls.json")
    )]

    for tradition, (G, data) in sorted(schemas.items()):
        inv = _ci(tradition, G)
        real_depth = inv["depth"]
        real_width = inv["width"]
        is_chain = inv["is_linear_chain"]

        p_depth = round(sum(1 for v in pooled_depths if v <= real_depth) / len(pooled_depths), 3)
        p_width = round(sum(1 for v in pooled_widths if v <= real_width) / len(pooled_widths), 3)

        chain_str = "yes" if is_chain else "no"
        print(
            f"  {tradition:<30} {real_depth:>6} {p_depth:>6}  "
            f"{real_width:>6} {p_width:>6}  {chain_str:>6}"
        )
    _sep()
    print("  p(<=): proportion of pooled controls with value <= the real schema's value")
    print("  p(<=) near 1.0 = the real schema is unusually deep/wide relative to null")
    print("  p(<=) near 0.0 = the real schema is unusually shallow/narrow relative to null")

    print()
    _sep("=")
    print(f"  Output: {os.path.abspath(CONTROLS_DIR)}")
    _sep("=")
    print()


def _print_stats_row(stats: dict) -> None:
    for metric in METRICS:
        s = stats.get(metric, {})
        if s:
            print(f"  {metric:<22}  mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"p05={s['p05']:.2f}  p50={s['p50']:.2f}  p95={s['p95']:.2f}")
    chain_pct = round(stats.get("linear_chain_frequency", 0) * 100, 1)
    print(f"  {'linear_chain_pct':<22}  {chain_pct}%")


def _load(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()

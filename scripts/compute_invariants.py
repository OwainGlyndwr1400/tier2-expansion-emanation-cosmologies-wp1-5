"""
compute_invariants.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Step 2: Compute topological invariants for each loaded schema.

Invariants computed
-------------------
Structural:
  - node_count, edge_count
  - depth (longest path root->leaf, in edges)
  - diameter (longest shortest path between any two nodes)
  - max_branching_factor, mean_branching_factor
  - leaf_count, branching_node_count
  - width (max nodes at any single depth level)

Distribution:
  - depth_profile         {level: node_count}
  - in_degree_sequence    sorted list of in-degrees
  - out_degree_sequence   sorted list of out-degrees
  - edge_type_counts      {type: count}
  - functional_role_counts {role: count}

Centrality:
  - betweenness_centrality  {node_id: score}
  - closeness_centrality    {node_id: score}

Canonical sequences (for isomorphism comparison):
  - role_sequence_by_level  {level: [functional_role, ...]}
  - topological_sort        ordered list of node IDs
  - wl_hash                 Weisfeiler-Leman graph hash (structure only)
  - wl_hash_with_roles      WL hash using functional_role as node label

Outputs
-------
  outputs/invariants/{tradition}_invariants.json  -- per-tradition
  outputs/invariants/all_invariants.json          -- combined

Usage:
    python compute_invariants.py
"""

import json
import os
import sys
import networkx as nx

# Resolve path to encode_schemas.py in the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from encode_schemas import load_all_schemas, SCHEMAS_DIR, VALID_EDGE_TYPES, VALID_FUNCTIONAL_ROLES

OUTPUTS_DIR = os.path.join(_HERE, "..", "outputs", "invariants")


# ---------------------------------------------------------------------------
# Per-invariant computation functions
# ---------------------------------------------------------------------------

def _root(G: nx.DiGraph) -> str:
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    return roots[0] if roots else None


def compute_depth_profile(G: nx.DiGraph) -> dict[int, list[str]]:
    """
    Map each node to its level (shortest path from root).
    Returns {level: [node_id, ...]} ordered from root downward.
    """
    root = _root(G)
    if root is None:
        return {}
    lengths = nx.single_source_shortest_path_length(G, root)
    profile: dict[int, list[str]] = {}
    for node, level in lengths.items():
        profile.setdefault(level, []).append(node)
    return dict(sorted(profile.items()))


def compute_role_sequence_by_level(G: nx.DiGraph) -> dict[int, list[str]]:
    """
    Map each depth level to the sorted list of functional roles present.
    The key structure for cross-tradition comparison.
    """
    profile = compute_depth_profile(G)
    return {
        level: sorted(G.nodes[n]["functional_role"] for n in nodes)
        for level, nodes in profile.items()
    }


def compute_centrality(G: nx.DiGraph) -> tuple[dict, dict]:
    """
    Returns (betweenness, closeness) centrality dicts keyed by node ID.
    Scores rounded to 4 decimal places.
    """
    bc = nx.betweenness_centrality(G, normalized=True)
    cc = nx.closeness_centrality(G)
    return (
        {n: round(v, 4) for n, v in bc.items()},
        {n: round(v, 4) for n, v in cc.items()},
    )


def compute_wl_hash(G: nx.DiGraph) -> tuple[str, str]:
    """
    Returns:
        (wl_hash_structure, wl_hash_with_roles)

    Structure hash: treats all nodes as identical (topology only).
    Role hash: uses functional_role as the node label.
    """
    # Structure only
    h_struct = nx.weisfeiler_lehman_graph_hash(G)

    # With functional roles as labels
    # Build a copy with string labels for WL
    G_labeled = nx.DiGraph()
    for n, attrs in G.nodes(data=True):
        G_labeled.add_node(n, label=attrs.get("functional_role", "unknown"))
    for u, v in G.edges():
        G_labeled.add_edge(u, v)

    h_roles = nx.weisfeiler_lehman_graph_hash(G_labeled, node_attr="label")
    return h_struct, h_roles


def compute_invariants(tradition: str, G: nx.DiGraph) -> dict:
    """
    Compute the full invariant set for a single graph.
    Returns a flat + nested dict ready for JSON serialization.
    """
    root = _root(G)
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    branching_nodes = [n for n in G.nodes if G.out_degree(n) > 1]

    depth_profile = compute_depth_profile(G)
    role_seq = compute_role_sequence_by_level(G)

    # Width = max number of nodes at any single level
    width = max(len(nodes) for nodes in depth_profile.values()) if depth_profile else 0

    edge_types = [attrs.get("relationship", "") for _, _, attrs in G.edges(data=True)]
    edge_type_counts = {t: edge_types.count(t) for t in VALID_EDGE_TYPES if t in edge_types}

    roles = [attrs.get("functional_role", "") for _, attrs in G.nodes(data=True)]
    role_counts = {r: roles.count(r) for r in VALID_FUNCTIONAL_ROLES if r in roles}

    in_deg = sorted(d for _, d in G.in_degree())
    out_deg = sorted(d for _, d in G.out_degree())

    betweenness, closeness = compute_centrality(G)

    topo_sort = list(nx.topological_sort(G))

    h_struct, h_roles = compute_wl_hash(G)

    # Diameter: longest shortest path between any pair
    # In a DAG we use the underlying undirected graph for diameter
    # (directed diameter is the dag_longest_path_length for connected chains)
    dag_depth = nx.dag_longest_path_length(G)
    try:
        diameter = nx.diameter(G.to_undirected())
    except nx.NetworkXError:
        diameter = dag_depth  # fallback for disconnected (shouldn't happen post-validation)

    # Mean out-degree (branching factor) excluding leaves
    non_leaf_out = [G.out_degree(n) for n in G.nodes if G.out_degree(n) > 0]
    mean_branching = round(sum(non_leaf_out) / len(non_leaf_out), 4) if non_leaf_out else 0.0

    return {
        "tradition": tradition,

        # --- Structural ---
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "depth": dag_depth,
        "diameter": diameter,
        "width": width,
        "leaf_count": len(leaves),
        "branching_node_count": len(branching_nodes),
        "max_branching_factor": max(G.out_degree(n) for n in G.nodes) if G.number_of_nodes() > 0 else 0,
        "mean_branching_factor": mean_branching,
        "root": root,
        "leaves": leaves,
        "branching_nodes": branching_nodes,

        # --- Degree sequences ---
        "in_degree_sequence": in_deg,
        "out_degree_sequence": out_deg,

        # --- Distribution ---
        "edge_type_counts": edge_type_counts,
        "functional_role_counts": role_counts,

        # --- Depth profile ---
        "depth_profile": {str(k): v for k, v in depth_profile.items()},
        "role_sequence_by_level": {str(k): v for k, v in role_seq.items()},

        # --- Centrality (keyed by node ID) ---
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,

        # --- Canonical sequences ---
        "topological_sort": topo_sort,
        "wl_hash_structure": h_struct,
        "wl_hash_with_roles": h_roles,

        # --- Flags for fast screening ---
        "is_linear_chain": (len(branching_nodes) == 0),
        "has_fallen_node": "fallen" in role_counts,
        "has_process_nodes": "process" in role_counts,
        "has_fragmentation_edge": "fragmentation" in edge_type_counts,
        "has_contraction_edge": "contraction" in edge_type_counts,
        "has_reflection_edge": "reflection" in edge_type_counts,
    }


# ---------------------------------------------------------------------------
# Batch computation and output
# ---------------------------------------------------------------------------

def run_all(schemas_dir: str = SCHEMAS_DIR, outputs_dir: str = OUTPUTS_DIR) -> dict:
    """
    Compute invariants for all loaded schemas.
    Saves per-tradition JSONs and combined all_invariants.json.
    Returns {tradition: invariants_dict}.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    schemas = load_all_schemas(schemas_dir)
    all_invariants = {}

    for tradition, (G, _data) in schemas.items():
        inv = compute_invariants(tradition, G)
        all_invariants[tradition] = inv

        out_path = os.path.join(outputs_dir, f"{tradition}_invariants.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(inv, f, indent=2, ensure_ascii=False)

    combined_path = os.path.join(outputs_dir, "all_invariants.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_invariants, f, indent=2, ensure_ascii=False)

    return all_invariants


# ---------------------------------------------------------------------------
# CLI: compute and print comparison report
# ---------------------------------------------------------------------------

def _sep(char="-", width=72):
    print(char * width)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _sep("=")
    print("  WP 1.1 -- compute_invariants.py")
    print("  Topological Invariant Report")
    _sep("=")

    all_inv = run_all()
    traditions = sorted(all_inv.keys())

    # --- Summary table ---
    print()
    print("  STRUCTURAL SUMMARY")
    _sep()
    hdr = (
        f"  {'Tradition':<28} "
        f"{'N':>3} {'E':>3} {'D':>3} {'Diam':>4} "
        f"{'W':>3} {'Br':>4} {'Lv':>3}  Shape"
    )
    print(hdr)
    _sep()
    for t in traditions:
        inv = all_inv[t]
        shape = "chain" if inv["is_linear_chain"] else "tree"
        flags = []
        if inv["has_process_nodes"]:     flags.append("proc")
        if inv["has_fallen_node"]:       flags.append("fall")
        if inv["has_fragmentation_edge"]:flags.append("frag")
        if inv["has_contraction_edge"]:  flags.append("cont")
        if inv["has_reflection_edge"]:   flags.append("refl")
        shape_str = f"{shape}  [{'+'.join(flags)}]" if flags else shape
        print(
            f"  {t:<28} "
            f"{inv['node_count']:>3} {inv['edge_count']:>3} "
            f"{inv['depth']:>3} {inv['diameter']:>4} "
            f"{inv['width']:>3} {inv['mean_branching_factor']:>4} "
            f"{inv['leaf_count']:>3}  {shape_str}"
        )
    _sep()
    print("  N=nodes E=edges D=depth Diam=diameter W=width Br=mean-branch Lv=leaves")

    # --- Role sequence by level ---
    print()
    print("  FUNCTIONAL ROLE SEQUENCE BY LEVEL")
    _sep()
    for t in traditions:
        inv = all_inv[t]
        print(f"  {t}")
        for level, roles in sorted(inv["role_sequence_by_level"].items(), key=lambda x: int(x[0])):
            print(f"    L{level}: {', '.join(roles)}")
    _sep()

    # --- Betweenness centrality (top node per tradition) ---
    print()
    print("  PEAK BETWEENNESS CENTRALITY (highest-scoring node per tradition)")
    _sep()
    for t in traditions:
        inv = all_inv[t]
        bc = inv["betweenness_centrality"]
        if bc:
            top_node = max(bc, key=bc.get)
            top_score = bc[top_node]
            role = all_inv[t]["functional_role_counts"]  # we need the graph for this
            print(f"  {t:<28}  {top_node:<25} score={top_score:.4f}")
    _sep()

    # --- WL hashes ---
    print()
    print("  WEISFEILER-LEMAN HASHES (structural fingerprint)")
    _sep()
    print(f"  {'Tradition':<28}  {'WL-structure':<12}  WL-with-roles")
    _sep()
    for t in traditions:
        inv = all_inv[t]
        print(f"  {t:<28}  {inv['wl_hash_structure']:<12}  {inv['wl_hash_with_roles']}")
    _sep()

    # WL collision check
    struct_hashes = [all_inv[t]["wl_hash_structure"] for t in traditions]
    role_hashes = [all_inv[t]["wl_hash_with_roles"] for t in traditions]
    struct_unique = len(set(struct_hashes))
    role_unique = len(set(role_hashes))
    print(f"\n  WL-structure unique hashes: {struct_unique}/{len(traditions)}")
    print(f"  WL-with-roles unique hashes: {role_unique}/{len(traditions)}")
    if struct_unique < len(traditions):
        print("  [!] Structural hash collision detected -- exact VF2 test required for affected pair(s)")
    else:
        print("  All structural hashes distinct -- no exact isomorphism between any pair.")

    # --- Output confirmation ---
    print()
    _sep("=")
    print(f"  Invariants saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print(f"  Files: {len(traditions)} per-tradition JSONs + all_invariants.json")
    _sep("=")
    print()


if __name__ == "__main__":
    main()

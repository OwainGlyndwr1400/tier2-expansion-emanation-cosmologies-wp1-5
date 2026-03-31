"""
encode_schemas.py
WP 1.1 — Emanation Topology Analysis
Pipeline Step 1: Load and validate all tradition schemas as NetworkX DiGraphs.

Usage (standalone):
    python encode_schemas.py

Usage (import):
    from encode_schemas import load_all_schemas, load_schema, validate_dag

Author: Erydir-Ceisiwr / Lumos Research Agent
Date: 2026-03-28
"""

import json
import os
import sys
import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMAS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "schemas"
)

REQUIRED_NODE_ATTRS = {"id", "name", "functional_role", "level", "source_reference"}
VALID_FUNCTIONAL_ROLES = {
    "source", "first_emanation", "intellect", "soul",
    "intermediary", "fallen", "demiurge", "matter", "process"
}
VALID_EDGE_TYPES = {
    "emanation", "creation", "fragmentation", "contraction", "reflection",
    "succession"  # Added WP 1.3: theogonic displacement (Derveni Orphic)
}
NODE_COUNT_MIN = 4
NODE_COUNT_MAX = 20


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_schema(path: str) -> tuple[str, nx.DiGraph, dict]:
    """
    Load a single tradition JSON and return:
        (tradition_name, G, metadata)

    G is a NetworkX DiGraph with all node/edge attributes attached.
    metadata is the full parsed JSON (for access to topology, validation,
    alternative_encodings, etc. by downstream modules).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tradition = data.get("tradition", os.path.splitext(os.path.basename(path))[0])

    G = nx.DiGraph(tradition=tradition, source_file=path)

    # Add nodes with all JSON attributes as graph node attributes
    for node in data["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, **node)

    # Add edges with all attributes (including relationship)
    for edge in data["edges"]:
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        G.add_edge(edge["source"], edge["target"], **attrs)

    return tradition, G, data


# ---------------------------------------------------------------------------
# DAG contract validator (7 rules)
# ---------------------------------------------------------------------------

def validate_dag(G: nx.DiGraph, data: dict) -> tuple[bool, list[str]]:
    """
    Validate a loaded DiGraph against the 7-rule DAG contract.

    Returns:
        (passed: bool, errors: list[str])

    Rules:
        1. Acyclic — no cycles
        2. Single root — exactly one node with in-degree 0
        3. All connected — weakly connected (no isolated nodes)
        4. Node count 4–20 — within bounds
        5. All required attributes present — every node has the 5 required fields
        6. All edges labeled — every edge has a valid relationship type
        7. Functional roles valid — every node uses a canonical role
           (source_reference existence is implicitly covered by rule 5)
    """
    errors = []
    tradition = G.graph.get("tradition", "unknown")

    # Rule 1: Acyclic
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        errors.append(f"[{tradition}] FAIL Rule 1 (Acyclic): {len(cycles)} cycle(s) found: {cycles}")

    # Rule 2: Single root
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) != 1:
        errors.append(
            f"[{tradition}] FAIL Rule 2 (Single root): "
            f"found {len(roots)} root(s) — {roots}"
        )

    # Rule 3: All connected (weakly)
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        errors.append(
            f"[{tradition}] FAIL Rule 3 (Connected): "
            f"{len(components)} weakly connected components"
        )

    # Rule 4: Node count 4–20
    n = G.number_of_nodes()
    if not (NODE_COUNT_MIN <= n <= NODE_COUNT_MAX):
        errors.append(
            f"[{tradition}] FAIL Rule 4 (Node count): "
            f"{n} nodes (must be {NODE_COUNT_MIN}–{NODE_COUNT_MAX})"
        )

    # Rule 5: All required node attributes present
    for node_id, attrs in G.nodes(data=True):
        missing = REQUIRED_NODE_ATTRS - set(attrs.keys())
        if missing:
            errors.append(
                f"[{tradition}] FAIL Rule 5 (Attributes): "
                f"node '{node_id}' missing: {missing}"
            )

    # Rule 6: All edges have a valid labeled relationship
    for u, v, attrs in G.edges(data=True):
        rel = attrs.get("relationship", "")
        if not rel:
            errors.append(
                f"[{tradition}] FAIL Rule 6 (Edge label): "
                f"edge ({u} → {v}) has no relationship type"
            )
        elif rel not in VALID_EDGE_TYPES:
            errors.append(
                f"[{tradition}] FAIL Rule 6 (Edge label): "
                f"edge ({u} → {v}) has unknown type '{rel}' "
                f"(valid: {sorted(VALID_EDGE_TYPES)})"
            )

    # Rule 7: All functional roles are from the canonical vocabulary
    for node_id, attrs in G.nodes(data=True):
        role = attrs.get("functional_role", "")
        if role not in VALID_FUNCTIONAL_ROLES:
            errors.append(
                f"[{tradition}] FAIL Rule 7 (Functional roles): "
                f"node '{node_id}' has unknown role '{role}' "
                f"(valid: {sorted(VALID_FUNCTIONAL_ROLES)})"
            )

    passed = len(errors) == 0
    return passed, errors


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_all_schemas(schemas_dir: str = SCHEMAS_DIR) -> dict[str, tuple[nx.DiGraph, dict]]:
    """
    Load all *.json files from schemas_dir.

    Returns:
        {tradition_name: (G, metadata)}

    Raises:
        FileNotFoundError if schemas_dir does not exist.
        ValueError if any schema fails DAG validation.
    """
    if not os.path.isdir(schemas_dir):
        raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")

    json_files = sorted(
        f for f in os.listdir(schemas_dir) if f.endswith(".json")
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {schemas_dir}")

    schemas = {}
    all_errors = []

    for fname in json_files:
        path = os.path.join(schemas_dir, fname)
        tradition, G, data = load_schema(path)
        passed, errors = validate_dag(G, data)

        if not passed:
            all_errors.extend(errors)
        else:
            schemas[tradition] = (G, data)

    if all_errors:
        msg = "\n".join(all_errors)
        raise ValueError(f"Schema validation failed:\n{msg}")

    return schemas


# ---------------------------------------------------------------------------
# Topology summary helpers (used by downstream modules + __main__)
# ---------------------------------------------------------------------------

def compute_depth(G: nx.DiGraph) -> int:
    """Longest path length from root to any leaf (in edges)."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        return 0
    root = roots[0]
    lengths = nx.single_source_shortest_path_length(G, root)
    return max(lengths.values())


def compute_max_branching(G: nx.DiGraph) -> int:
    """Maximum out-degree across all nodes."""
    return max(dict(G.out_degree()).values()) if G.number_of_nodes() > 0 else 0


def topology_summary(tradition: str, G: nx.DiGraph) -> dict:
    """Return a flat dict of key topological invariants for quick comparison."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    edge_types = [attrs["relationship"] for _, _, attrs in G.edges(data=True)]
    roles = [attrs["functional_role"] for _, attrs in G.nodes(data=True)]

    return {
        "tradition": tradition,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "depth": compute_depth(G),
        "max_branching": compute_max_branching(G),
        "root": roots[0] if roots else None,
        "leaves": leaves,
        "edge_type_counts": {t: edge_types.count(t) for t in VALID_EDGE_TYPES if t in edge_types},
        "functional_role_counts": {r: roles.count(r) for r in VALID_FUNCTIONAL_ROLES if r in roles},
        "has_process_nodes": "process" in roles,
        "has_fallen_node": "fallen" in roles,
        "has_fragmentation_edge": "fragmentation" in edge_types,
        "has_contraction_edge": "contraction" in edge_types,
        "has_succession_edge": "succession" in edge_types,
    }


# ---------------------------------------------------------------------------
# CLI: validate all schemas and print report
# ---------------------------------------------------------------------------

def _print_separator(char="-", width=60):
    print(char * width)


def main():
    # Ensure UTF-8 output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _print_separator("=")
    print("  WP 1.1 -- encode_schemas.py")
    print("  DAG Contract Validation Report")
    _print_separator("=")

    schemas_dir = SCHEMAS_DIR
    json_files = sorted(f for f in os.listdir(schemas_dir) if f.endswith(".json"))

    all_passed = True
    summaries = []

    for fname in json_files:
        path = os.path.join(schemas_dir, fname)
        tradition, G, data = load_schema(path)
        passed, errors = validate_dag(G, data)

        status = "PASS [OK]" if passed else "FAIL [!!]"
        print(f"\n  {status}  {tradition}")
        _print_separator()

        if errors:
            all_passed = False
            for e in errors:
                print(f"    {e}")
        else:
            s = topology_summary(tradition, G)
            print(f"    Nodes          : {s['nodes']}")
            print(f"    Edges          : {s['edges']}")
            print(f"    Depth          : {s['depth']}")
            print(f"    Max branching  : {s['max_branching']}")
            print(f"    Root           : {s['root']}")
            print(f"    Leaves         : {', '.join(s['leaves'])}")
            print(f"    Edge types     : {s['edge_type_counts']}")
            print(f"    Functional roles: {s['functional_role_counts']}")
            flags = []
            if s["has_process_nodes"]:   flags.append("process-nodes")
            if s["has_fallen_node"]:     flags.append("fallen-node")
            if s["has_fragmentation_edge"]: flags.append("fragmentation")
            if s["has_contraction_edge"]:   flags.append("contraction")
            if s["has_succession_edge"]:    flags.append("succession")
            if flags:
                print(f"    Distinctive    : {', '.join(flags)}")
            summaries.append(s)

    print()
    _print_separator("=")
    if all_passed:
        print(f"  ALL {len(json_files)} SCHEMAS PASSED -- pipeline is clear to proceed.")
    else:
        print(f"  VALIDATION ERRORS DETECTED -- fix before running pipeline.")
    _print_separator("=")

    if summaries:
        print("\n  CROSS-TRADITION TOPOLOGY TABLE")
        _print_separator()
        hdr = f"  {'Tradition':<30} {'N':>3} {'E':>3} {'D':>3} {'Br':>3}  Flags"
        print(hdr)
        _print_separator()
        for s in summaries:
            flags = []
            if s["has_process_nodes"]:      flags.append("proc")
            if s["has_fallen_node"]:        flags.append("fall")
            if s["has_fragmentation_edge"]: flags.append("frag")
            if s["has_contraction_edge"]:   flags.append("cont")
            if s["has_succession_edge"]:    flags.append("succ")
            flag_str = "+".join(flags) if flags else "—"
            print(
                f"  {s['tradition']:<30} "
                f"{s['nodes']:>3} {s['edges']:>3} "
                f"{s['depth']:>3} {s['max_branching']:>3}  {flag_str}"
            )
        _print_separator()
        print()


if __name__ == "__main__":
    main()

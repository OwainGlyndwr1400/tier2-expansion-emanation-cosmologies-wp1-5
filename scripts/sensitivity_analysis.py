"""
sensitivity_analysis.py
WP 1.5 -- Sensitivity Analysis, Sub-Clustering & Methodological Extensions

Analyses:
1-4. (WP 1.4 inherited) Branching alternatives, Popol Vuh, Gospel of Mary direction, sub-clustering
5.   (WP 1.5) Edge-weighted GED sensitivity: 4 cost matrices (Primary, Uniform, Steep, Compressed)
6.   (WP 1.5) Direction testing for branching trees: reverse edges, check if GED > 0
7.   Summary comparison table

Usage:
    python sensitivity_analysis.py
"""

import json
import os
import sys
import copy
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_schemas import load_all_schemas, compute_depth, compute_max_branching

SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "schemas")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
GED_TIMEOUT = 30


def _sep(char="-", width=70):
    print(char * width)


# ---------------------------------------------------------------------------
# Branching alternative builders
# ---------------------------------------------------------------------------

def build_ishraq_branching(schemas):
    """Build Suhrawardi branching alternative: victorial_lights has out-degree 2."""
    G_orig, data = schemas["ishraq_illuminationist"]
    G = G_orig.copy()
    # Remove the single edge victorial_lights -> lords_of_species
    # Add two edges: victorial_lights -> lords_of_species AND victorial_lights -> accidental_lights
    if G.has_edge("victorial_lights", "lords_of_species"):
        G.remove_edge("lords_of_species", "accidental_lights")
        G.add_edge("victorial_lights", "accidental_lights",
                   relationship="emanation",
                   description="Branching alt: longitudinal chain directly generates accidental lights")
    return G


def build_popol_vuh_no_xibalba(schemas):
    """Build Popol Vuh without-Xibalba alternative: remove Hero Twin cycle.

    Removes xibalba_descent, hero_twins, sun_moon nodes and their edges.
    Adds direct edge wooden_people -> maize_discovery.
    Result: 8-node schema that should reclassify from branching tree to linear chain
    (creative_speech still has out-degree 4: earth_animals, mud, wood, maize_discovery).
    Actually remains branching (factor 4) even without Xibalba due to the 3 creation attempts.
    But the Xibalba side-branch is removed.
    """
    G_orig, data = schemas["popol_vuh_maya"]
    G = G_orig.copy()
    # Remove the Xibalba cycle nodes
    for node in ["xibalba_descent", "hero_twins", "sun_moon"]:
        if node in G:
            G.remove_node(node)
    # Add direct edge from wooden_people to maize_discovery
    if not G.has_edge("wooden_people", "maize_discovery"):
        G.add_edge("wooden_people", "maize_discovery",
                   relationship="creation",
                   description="Direct progression from failed Wood to Maize discovery (Xibalba omitted)")
    return G


def build_gospel_mary_ascending(schemas):
    """Build Gospel of Mary direction-sensitive alternative: reverse all edges.

    Changes edge types from emanation to contraction (soul ascending, not descending).
    Reverses direction: material_body becomes root, silence_rest becomes leaf.
    Topologically identical (linear chain, depth 8) but semantically inverted.
    """
    G_orig, data = schemas["gospel_of_mary"]
    G_rev = nx.DiGraph()
    # Copy all nodes
    for node, attrs in G_orig.nodes(data=True):
        G_rev.add_node(node, **attrs)
    # Reverse all edges and change type to contraction
    for u, v, attrs in G_orig.edges(data=True):
        new_attrs = dict(attrs)
        new_attrs["relationship"] = "contraction"
        new_attrs["description"] = f"Soul ascends: {v} -> {u} (reversed)"
        G_rev.add_edge(v, u, **new_attrs)
    return G_rev


def build_zurvanite_bundahishn(schemas):
    """Build Zurvanite alternative: Zurvan at apex, bifurcating to Ohrmazd + Ahriman."""
    G_orig, data = schemas["bundahishn_zoroastrian"]
    G = G_orig.copy()
    # Add Zurvan as new root
    G.add_node("zurvan", id="zurvan", name="Zurvan (Infinite Time)",
               functional_role="source", level=-1,
               source_reference="Zaehner 1955, Zurvan ch.3")
    # Add Ahriman as dark branch
    G.add_node("ahriman", id="ahriman", name="Ahriman (Angra Mainyu)",
               functional_role="demiurge", level=0,
               source_reference="GBd 1.1-1.6; Zaehner 1955")
    # Reclassify Ohrmazd from source to first_emanation
    G.nodes["ohrmazd"]["functional_role"] = "first_emanation"
    G.nodes["ohrmazd"]["level"] = 0
    # Add edges from Zurvan
    G.add_edge("zurvan", "ohrmazd", relationship="emanation",
               description="Ohrmazd born from Zurvan's sacrifice")
    G.add_edge("zurvan", "ahriman", relationship="fragmentation",
               description="Ahriman born from Zurvan's doubt")
    return G


# ---------------------------------------------------------------------------
# GED computation
# ---------------------------------------------------------------------------

def pairwise_ged(graphs: dict) -> dict:
    """Compute pairwise structural GED for all pairs."""
    names = sorted(graphs.keys())
    matrix = {}
    n = len(names)
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            ged = nx.graph_edit_distance(graphs[a], graphs[b], timeout=GED_TIMEOUT)
            if ged is None:
                ged = float("inf")
            matrix[(a, b)] = ged
            matrix[(b, a)] = ged
            done += 1
            if done % 10 == 0:
                print(f"    GED: {done}/{total} pairs computed...")
    for name in names:
        matrix[(name, name)] = 0.0
    return matrix


# ---------------------------------------------------------------------------
# Topology family classification
# ---------------------------------------------------------------------------

def classify_families(graphs: dict):
    """Classify into linear_chain vs branching_tree by max branching factor."""
    families = {"linear_chain": [], "branching_tree": []}
    for name, G in graphs.items():
        mb = compute_max_branching(G)
        if mb <= 1:
            families["linear_chain"].append(name)
        else:
            families["branching_tree"].append(name)
    return families


def family_stats(families, ged_matrix, all_names):
    """Compute intra-family and inter-family mean GED."""
    def mean_ged(members):
        if len(members) < 2:
            return 0.0
        geds = [ged_matrix[(a, b)] for a, b in combinations(members, 2)]
        return np.mean(geds)

    chains = families.get("linear_chain", [])
    branches = families.get("branching_tree", [])

    intra_chain = mean_ged(chains)
    intra_branch = mean_ged(branches)

    # Inter-family
    inter_geds = [ged_matrix[(a, b)] for a in chains for b in branches]
    inter = np.mean(inter_geds) if inter_geds else 0.0

    # Separation ratio
    min_intra = min(intra_chain, intra_branch) if (intra_chain > 0 and intra_branch > 0) else max(intra_chain, intra_branch)
    sep_ratio = inter / min_intra if min_intra > 0 else float("inf")

    return {
        "intra_chain": round(intra_chain, 4),
        "intra_branch": round(intra_branch, 4),
        "inter_family": round(inter, 4),
        "separation_ratio": round(sep_ratio, 4),
        "n_chains": len(chains),
        "n_branches": len(branches),
    }


# ---------------------------------------------------------------------------
# Sub-clustering analysis
# ---------------------------------------------------------------------------

def subclustering_analysis(schemas, ged_matrix):
    """
    Sub-clustering within the linear-chain family by node count.
    Tests whether {5-node chains} and {7-8-node chains} form distinct sub-clusters.
    """
    print("\n  SUB-CLUSTERING WITHIN LINEAR-CHAIN FAMILY")
    _sep()

    # Get linear chains
    chains = []
    for name, (G, _) in schemas.items():
        if compute_max_branching(G) <= 1:
            chains.append((name, G.number_of_nodes()))

    chains.sort(key=lambda x: x[1])
    print(f"  Linear chains ({len(chains)} members):")
    for name, n in chains:
        print(f"    {name:<35} N={n}")

    # Define sub-clusters by node count
    small = [name for name, n in chains if n <= 6]  # 5-6 nodes
    large = [name for name, n in chains if n >= 7]   # 7-8 nodes

    print(f"\n  Sub-cluster A (N<=6, 'compact'): {small}")
    print(f"  Sub-cluster B (N>=7, 'deep'):    {large}")

    # Intra-cluster GED
    def mean_ged(members):
        if len(members) < 2:
            return 0.0
        geds = [ged_matrix[(a, b)] for a, b in combinations(members, 2)]
        return np.mean(geds)

    intra_small = mean_ged(small)
    intra_large = mean_ged(large)
    inter_geds = [ged_matrix[(a, b)] for a in small for b in large]
    inter = np.mean(inter_geds) if inter_geds else 0.0

    print(f"\n  Intra-cluster A (compact) mean GED: {intra_small:.4f}")
    print(f"  Intra-cluster B (deep) mean GED:    {intra_large:.4f}")
    print(f"  Inter-cluster mean GED:             {inter:.4f}")

    min_intra = min(intra_small, intra_large) if (intra_small > 0 and intra_large > 0) else max(intra_small, intra_large)
    sub_sep = inter / min_intra if min_intra > 0 else float("inf")
    print(f"  Sub-cluster separation ratio:       {sub_sep:.4f}x")

    # Silhouette-style score
    silhouettes = []
    all_members = small + large
    for member in all_members:
        own_cluster = small if member in small else large
        other_cluster = large if member in small else small

        if len(own_cluster) > 1:
            a_i = np.mean([ged_matrix[(member, m)] for m in own_cluster if m != member])
        else:
            a_i = 0.0

        if other_cluster:
            b_i = np.mean([ged_matrix[(member, m)] for m in other_cluster])
        else:
            b_i = 0.0

        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        silhouettes.append(s_i)

    mean_silhouette = np.mean(silhouettes)
    print(f"\n  Mean silhouette score: {mean_silhouette:.4f}")
    print(f"  (>0.5 = strong clustering, >0.25 = moderate, <0.25 = weak)")
    _sep()

    return {
        "small_cluster": small,
        "large_cluster": large,
        "intra_small": round(intra_small, 4),
        "intra_large": round(intra_large, 4),
        "inter_cluster": round(inter, 4),
        "sub_separation_ratio": round(sub_sep, 4),
        "mean_silhouette": round(mean_silhouette, 4),
        "silhouettes": {m: round(s, 4) for m, s in zip(all_members, silhouettes)},
    }


# ---------------------------------------------------------------------------
# WP 1.5: Edge-weighted GED sensitivity (4 cost matrices)
# ---------------------------------------------------------------------------

# Cost matrices: {(type_a, type_b): cost}
# Same type always = 0.0. Only off-diagonal costs differ.
COST_MATRICES = {
    "primary": {
        # WP 1.4 original: same=0, related=0.5, unrelated=1.0, opposite=1.5
        ("emanation", "creation"): 0.5, ("creation", "emanation"): 0.5,
        ("fragmentation", "contraction"): 0.5, ("contraction", "fragmentation"): 0.5,
        ("reflection", "succession"): 0.5, ("succession", "reflection"): 0.5,
        ("creation", "fragmentation"): 1.5, ("fragmentation", "creation"): 1.5,
        ("emanation", "contraction"): 1.5, ("contraction", "emanation"): 1.5,
    },
    "uniform": {},  # All mismatches cost 1.0 (no related/opposite distinction)
    "steep": {
        # Related=0.3, opposite=2.0 (wider spread)
        ("emanation", "creation"): 0.3, ("creation", "emanation"): 0.3,
        ("fragmentation", "contraction"): 0.3, ("contraction", "fragmentation"): 0.3,
        ("reflection", "succession"): 0.3, ("succession", "reflection"): 0.3,
        ("creation", "fragmentation"): 2.0, ("fragmentation", "creation"): 2.0,
        ("emanation", "contraction"): 2.0, ("contraction", "emanation"): 2.0,
    },
    "compressed": {
        # Related=0.8, opposite=1.2 (narrow spread)
        ("emanation", "creation"): 0.8, ("creation", "emanation"): 0.8,
        ("fragmentation", "contraction"): 0.8, ("contraction", "fragmentation"): 0.8,
        ("reflection", "succession"): 0.8, ("succession", "reflection"): 0.8,
        ("creation", "fragmentation"): 1.2, ("fragmentation", "creation"): 1.2,
        ("emanation", "contraction"): 1.2, ("contraction", "emanation"): 1.2,
    },
}


def _make_edge_cost_fn(cost_dict):
    """Create an edge substitution cost function from a cost dictionary."""
    def edge_subst_cost(attrs1, attrs2):
        r1 = attrs1.get("relationship", "")
        r2 = attrs2.get("relationship", "")
        if r1 == r2:
            return 0.0
        return cost_dict.get((r1, r2), 1.0)
    return edge_subst_cost


def _role_subst_cost(attrs1, attrs2):
    """Node substitution cost: 0 if same functional_role, 1 otherwise."""
    return 0.0 if attrs1.get("functional_role") == attrs2.get("functional_role") else 1.0


def weighted_ged_sensitivity(schemas):
    """
    WP 1.5 Analysis: Test whether edge-weighted GED results are robust
    across 4 alternative cost matrices.

    Key question: Do all structural isomorphisms (GED=0 pairs) get resolved
    (weighted GED > 0) under every cost matrix?
    """
    print("\n\n  WP 1.5 ANALYSIS: EDGE-WEIGHTED GED SENSITIVITY")
    _sep("=")

    graphs = {name: G for name, (G, _) in schemas.items()}
    names = sorted(graphs.keys())
    pairs = list(combinations(names, 2))

    # First find structural isomorphism pairs (GED = 0)
    # Load from pre-computed matrix
    ged_path = os.path.join(OUTPUTS_DIR, "similarity_matrix", "structural_ged.json")
    iso_pairs = []
    if os.path.exists(ged_path):
        with open(ged_path, "r") as f:
            ged_raw = json.load(f)
        for key, val_dict in ged_raw.items():
            parts = key.split(" vs ")
            if len(parts) == 2:
                sged = val_dict.get("structural_ged", None)
                if sged is not None and sged == 0.0:
                    iso_pairs.append(tuple(parts))

    if iso_pairs:
        print(f"  Structural isomorphism pairs (GED=0): {len(iso_pairs)}")
        for a, b in iso_pairs:
            print(f"    {a} vs {b}")
    else:
        print("  No structural isomorphism pairs found (all GED > 0).")
        print("  Running full weighted GED on all pairs instead.")

    # Test each cost matrix
    results = {}
    test_pairs = iso_pairs if iso_pairs else pairs[:20]  # If no iso pairs, test a sample

    for matrix_name, cost_dict in COST_MATRICES.items():
        print(f"\n  Cost matrix: {matrix_name}")
        edge_fn = _make_edge_cost_fn(cost_dict)
        matrix_results = {}
        all_resolved = True

        for a, b in test_pairs:
            Ga, Gb = graphs[a], graphs[b]
            wged = nx.graph_edit_distance(
                Ga, Gb,
                node_subst_cost=_role_subst_cost,
                edge_subst_cost=edge_fn,
                timeout=GED_TIMEOUT,
            )
            if wged is None:
                wged = float("inf")
            matrix_results[f"{a} vs {b}"] = round(wged, 4)
            if wged == 0.0 and (a, b) in iso_pairs:
                all_resolved = False
            print(f"    {a} vs {b}: wGED = {wged:.4f}")

        results[matrix_name] = {
            "pairs": matrix_results,
            "all_isomorphisms_resolved": all_resolved,
        }

        if iso_pairs:
            status = "ALL RESOLVED" if all_resolved else "SOME UNRESOLVED"
            print(f"  --> Isomorphism resolution: {status}")

    # Summary table
    print(f"\n  WEIGHTED GED SENSITIVITY SUMMARY")
    _sep()
    print(f"  {'Pair':<55} {'Primary':>8} {'Uniform':>8} {'Steep':>8} {'Compr':>8}")
    _sep()
    for a, b in test_pairs:
        key = f"{a} vs {b}"
        vals = [results[m]["pairs"].get(key, "N/A") for m in ["primary", "uniform", "steep", "compressed"]]
        val_strs = [f"{v:>8.2f}" if isinstance(v, float) else f"{'N/A':>8}" for v in vals]
        print(f"  {key:<55} {'  '.join(val_strs)}")
    _sep()

    # Check robustness: are all isomorphisms resolved under all matrices?
    all_robust = all(r["all_isomorphisms_resolved"] for r in results.values())
    print(f"\n  Robustness: {'ALL matrices resolve all isomorphisms' if all_robust else 'SOME matrices fail to resolve isomorphisms'}")

    return results


# ---------------------------------------------------------------------------
# WP 1.5: Direction testing for branching trees
# ---------------------------------------------------------------------------

def direction_test_branching_trees(schemas):
    """
    WP 1.5 Analysis: For each branching tree, reverse all edges and compute
    GED to the original. A non-zero GED indicates direction matters.

    Prediction from handoff: GED > 0 for asymmetric trees (Valentinian, Manichaean).
    GED = 0 for symmetric trees (Genesis).
    """
    print("\n\n  WP 1.5 ANALYSIS: DIRECTION TESTING FOR BRANCHING TREES")
    _sep("=")

    graphs = {name: G for name, (G, _) in schemas.items()}
    branching = [name for name, G in graphs.items() if compute_max_branching(G) > 1]

    print(f"  Branching trees to test: {len(branching)}")
    results = {}

    for name in sorted(branching):
        G = graphs[name]
        # Build reversed graph
        G_rev = nx.DiGraph()
        for node, attrs in G.nodes(data=True):
            G_rev.add_node(node, **attrs)
        for u, v, attrs in G.edges(data=True):
            G_rev.add_edge(v, u, **attrs)

        # Structural GED (original vs reversed)
        ged = nx.graph_edit_distance(G, G_rev, timeout=GED_TIMEOUT)
        if ged is None:
            ged = float("inf")

        # Role-labeled GED
        role_ged = nx.graph_edit_distance(
            G, G_rev,
            node_subst_cost=_role_subst_cost,
            timeout=GED_TIMEOUT,
        )
        if role_ged is None:
            role_ged = float("inf")

        direction_matters = ged > 0.0
        n_nodes = G.number_of_nodes()
        mb = compute_max_branching(G)

        results[name] = {
            "structural_ged_vs_reversed": round(ged, 4),
            "role_ged_vs_reversed": round(role_ged, 4) if role_ged != float("inf") else "timeout",
            "direction_matters": direction_matters,
            "nodes": n_nodes,
            "max_branching": mb,
        }

        status = "DIRECTION MATTERS" if direction_matters else "direction invariant"
        print(f"  {name:<35} sGED={ged:.1f}  rGED={role_ged:.1f}  [{status}]  (N={n_nodes}, Br={mb})")

    # Summary
    n_matters = sum(1 for r in results.values() if r["direction_matters"])
    n_invariant = len(results) - n_matters
    print(f"\n  Direction matters: {n_matters}/{len(results)} branching trees")
    print(f"  Direction invariant: {n_invariant}/{len(results)} branching trees")

    if n_matters > 0:
        print("  --> Direction is NOT universally invariant for branching trees.")
        print("  --> This contrasts with linear chains (validated as direction-invariant in WP 1.4).")
    else:
        print("  --> Direction is universally invariant for branching trees (same as linear chains).")

    _sep()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _sep("=")
    print("  WP 1.5 -- Sensitivity Analysis, Sub-Clustering & Methodological Extensions")
    _sep("=")

    # Load primary schemas
    schemas = load_all_schemas(SCHEMAS_DIR)
    primary_graphs = {name: G for name, (G, _) in schemas.items()}

    # -----------------------------------------------------------------------
    # ANALYSIS 1: Primary corpus family classification (already done by pipeline,
    # but we recompute here for the sensitivity comparison baseline)
    # -----------------------------------------------------------------------
    print("\n  BASELINE: Primary Corpus (22 schemas)")
    _sep()

    # Load pre-computed GED matrix if available
    ged_path = os.path.join(OUTPUTS_DIR, "similarity_matrix", "structural_ged.json")
    if os.path.exists(ged_path):
        with open(ged_path, "r") as f:
            ged_raw = json.load(f)
        # Convert "A vs B" keyed dict to tuple-keyed dict
        primary_ged = {}
        names = sorted(primary_graphs.keys())
        for key, val_dict in ged_raw.items():
            parts = key.split(" vs ")
            if len(parts) == 2:
                a, b = parts
                ged_val = val_dict.get("structural_ged", 0.0)
                primary_ged[(a, b)] = ged_val
                primary_ged[(b, a)] = ged_val
        # Self-distances
        for name in names:
            primary_ged[(name, name)] = 0.0
        print("  Loaded pre-computed GED matrix.")
    else:
        print("  Computing pairwise GED (may take a few minutes)...")
        primary_ged = pairwise_ged(primary_graphs)

    primary_families = classify_families(primary_graphs)
    primary_stats = family_stats(primary_families, primary_ged, list(primary_graphs.keys()))

    print(f"  Linear chains ({primary_stats['n_chains']}): {', '.join(sorted(primary_families['linear_chain']))}")
    print(f"  Branching trees ({primary_stats['n_branches']}): {', '.join(sorted(primary_families['branching_tree']))}")
    print(f"  Intra-chain GED:  {primary_stats['intra_chain']}")
    print(f"  Intra-branch GED: {primary_stats['intra_branch']}")
    print(f"  Inter-family GED: {primary_stats['inter_family']}")
    print(f"  Separation ratio: {primary_stats['separation_ratio']}x")

    # -----------------------------------------------------------------------
    # ANALYSIS 2: Suhrawardi branching alternative
    # -----------------------------------------------------------------------
    print("\n\n  SENSITIVITY 1: Suhrawardi Branching Alternative")
    _sep()
    print("  Reclassifying ishraq_illuminationist from linear chain to branching tree.")
    print("  (Victorial Lights gets out-degree 2: -> Lords of Species AND -> Accidental Lights)")

    alt_ishraq = build_ishraq_branching(schemas)
    alt1_graphs = dict(primary_graphs)
    alt1_graphs["ishraq_illuminationist"] = alt_ishraq

    print("  Computing GED for affected pairs...")
    # Only recompute pairs involving Ishraq
    alt1_ged = dict(primary_ged)
    for other in alt1_graphs:
        if other == "ishraq_illuminationist":
            continue
        ged_val = nx.graph_edit_distance(alt_ishraq, alt1_graphs[other], timeout=GED_TIMEOUT)
        if ged_val is None:
            ged_val = float("inf")
        alt1_ged[("ishraq_illuminationist", other)] = ged_val
        alt1_ged[(other, "ishraq_illuminationist")] = ged_val

    alt1_families = classify_families(alt1_graphs)
    alt1_stats = family_stats(alt1_families, alt1_ged, list(alt1_graphs.keys()))

    print(f"  Linear chains ({alt1_stats['n_chains']}): {', '.join(sorted(alt1_families['linear_chain']))}")
    print(f"  Branching trees ({alt1_stats['n_branches']}): {', '.join(sorted(alt1_families['branching_tree']))}")
    print(f"  Intra-chain GED:  {alt1_stats['intra_chain']}")
    print(f"  Intra-branch GED: {alt1_stats['intra_branch']}")
    print(f"  Inter-family GED: {alt1_stats['inter_family']}")
    print(f"  Separation ratio: {alt1_stats['separation_ratio']}x")

    # Key isomorphism check
    ishraq_samkhya = alt1_ged.get(("ishraq_illuminationist", "samkhya"), "N/A")
    print(f"\n  Ishraq-Samkhya GED under branching: {ishraq_samkhya}")
    print(f"  (Was 0.0 under primary linear encoding)")
    if ishraq_samkhya != 0.0:
        print("  --> Ishraq-Samkhya isomorphism DISSOLVED by branching alternative.")
    else:
        print("  --> Ishraq-Samkhya isomorphism PRESERVED even under branching.")

    # -----------------------------------------------------------------------
    # ANALYSIS 3: Zurvanite Bundahishn alternative
    # -----------------------------------------------------------------------
    print("\n\n  SENSITIVITY 2: Zurvanite Bundahishn Alternative")
    _sep()
    print("  Adding Zurvan at apex; bifurcating to Ohrmazd + Ahriman.")
    print("  Reclassifying bundahishn_zoroastrian from linear chain to branching tree.")

    alt_bundahishn = build_zurvanite_bundahishn(schemas)
    alt2_graphs = dict(primary_graphs)
    alt2_graphs["bundahishn_zoroastrian"] = alt_bundahishn

    print("  Computing GED for affected pairs...")
    alt2_ged = dict(primary_ged)
    for other in alt2_graphs:
        if other == "bundahishn_zoroastrian":
            continue
        ged_val = nx.graph_edit_distance(alt_bundahishn, alt2_graphs[other], timeout=GED_TIMEOUT)
        if ged_val is None:
            ged_val = float("inf")
        alt2_ged[("bundahishn_zoroastrian", other)] = ged_val
        alt2_ged[(other, "bundahishn_zoroastrian")] = ged_val

    alt2_families = classify_families(alt2_graphs)
    alt2_stats = family_stats(alt2_families, alt2_ged, list(alt2_graphs.keys()))

    print(f"  Linear chains ({alt2_stats['n_chains']}): {', '.join(sorted(alt2_families['linear_chain']))}")
    print(f"  Branching trees ({alt2_stats['n_branches']}): {', '.join(sorted(alt2_families['branching_tree']))}")
    print(f"  Intra-chain GED:  {alt2_stats['intra_chain']}")
    print(f"  Intra-branch GED: {alt2_stats['intra_branch']}")
    print(f"  Inter-family GED: {alt2_stats['inter_family']}")
    print(f"  Separation ratio: {alt2_stats['separation_ratio']}x")

    # Key comparison
    bund_ishraq = alt2_ged.get(("bundahishn_zoroastrian", "ishraq_illuminationist"), "N/A")
    print(f"\n  Zurvanite Bundahishn vs Ishraq GED: {bund_ishraq}")
    print(f"  (Was 2.0 under primary encoding)")

    # -----------------------------------------------------------------------
    # ANALYSIS 3b: Popol Vuh without-Xibalba alternative (WP 1.4)
    # -----------------------------------------------------------------------
    print("\n\n  SENSITIVITY 3: Popol Vuh Without-Xibalba Alternative")
    _sep()
    print("  Removing Hero Twin / Xibalba cycle (3 nodes, 3 edges).")
    print("  Tests whether Popol Vuh remains branching tree without the side-branch.")

    alt_popol = build_popol_vuh_no_xibalba(schemas)
    alt3_graphs = dict(primary_graphs)
    alt3_graphs["popol_vuh_maya"] = alt_popol

    alt3_families = classify_families(alt3_graphs)
    popol_class = "branching_tree" if "popol_vuh_maya" in alt3_families.get("branching_tree", []) else "linear_chain"
    popol_nodes = alt_popol.number_of_nodes()
    popol_mb = compute_max_branching(alt_popol)

    print(f"  Without-Xibalba: {popol_nodes} nodes, max_branching={popol_mb}")
    print(f"  Classification: {popol_class}")
    if popol_class == "branching_tree":
        print("  --> Still branching tree (creative_speech has out-degree > 1 from 3 creation attempts).")
        print("  --> Xibalba removal does NOT change family assignment.")
    else:
        print("  --> Reclassified to linear chain! Xibalba was the sole source of branching.")
        print("  --> This weakens the geographic generalisation claim.")

    # Compute GED for affected pairs
    print("  Computing GED for affected pairs...")
    alt3_ged = dict(primary_ged)
    for other in alt3_graphs:
        if other == "popol_vuh_maya":
            continue
        ged_val = nx.graph_edit_distance(alt_popol, alt3_graphs[other], timeout=GED_TIMEOUT)
        if ged_val is None:
            ged_val = float("inf")
        alt3_ged[("popol_vuh_maya", other)] = ged_val
        alt3_ged[(other, "popol_vuh_maya")] = ged_val
    alt3_ged[("popol_vuh_maya", "popol_vuh_maya")] = 0.0

    alt3_full_families = classify_families(alt3_graphs)
    alt3_stats = family_stats(alt3_full_families, alt3_ged, list(alt3_graphs.keys()))

    print(f"  Linear chains ({alt3_stats['n_chains']}): {', '.join(sorted(alt3_full_families['linear_chain']))}")
    print(f"  Branching trees ({alt3_stats['n_branches']}): {', '.join(sorted(alt3_full_families['branching_tree']))}")
    print(f"  Separation ratio: {alt3_stats['separation_ratio']}x")

    # -----------------------------------------------------------------------
    # ANALYSIS 3c: Gospel of Mary direction-sensitive (WP 1.4)
    # -----------------------------------------------------------------------
    print("\n\n  SENSITIVITY 4: Gospel of Mary Direction-Sensitive (Ascending)")
    _sep()
    print("  Reversing all edges; changing emanation -> contraction.")
    print("  Tests whether direction affects GED comparisons.")

    alt_gospel = build_gospel_mary_ascending(schemas)
    alt4_graphs = dict(primary_graphs)
    alt4_graphs["gospel_of_mary"] = alt_gospel

    # Compute GED for affected pairs
    print("  Computing GED for affected pairs...")
    alt4_ged = dict(primary_ged)
    for other in alt4_graphs:
        if other == "gospel_of_mary":
            continue
        ged_val = nx.graph_edit_distance(alt_gospel, alt4_graphs[other], timeout=GED_TIMEOUT)
        if ged_val is None:
            ged_val = float("inf")
        alt4_ged[("gospel_of_mary", other)] = ged_val
        alt4_ged[(other, "gospel_of_mary")] = ged_val
    alt4_ged[("gospel_of_mary", "gospel_of_mary")] = 0.0

    alt4_families = classify_families(alt4_graphs)
    gospel_class = "linear_chain" if "gospel_of_mary" in alt4_families.get("linear_chain", []) else "branching_tree"
    print(f"  Classification under reversal: {gospel_class}")

    # Compare GED values: original vs reversed
    n_changed = 0
    n_total = 0
    for other in primary_graphs:
        if other == "gospel_of_mary":
            continue
        orig = primary_ged.get(("gospel_of_mary", other), None)
        rev = alt4_ged.get(("gospel_of_mary", other), None)
        if orig is not None and rev is not None:
            n_total += 1
            if abs(orig - rev) > 0.01:
                n_changed += 1
                print(f"  CHANGED: gospel_of_mary vs {other}: {orig} -> {rev}")

    if n_changed == 0:
        print(f"  --> ALL {n_total} GED values UNCHANGED under direction reversal.")
        print("  --> Direction-agnostic approach is validated: topology is direction-invariant.")
    else:
        print(f"  --> {n_changed}/{n_total} GED values changed under direction reversal.")
        print("  --> Direction matters for structural comparison. Note in paper.")

    # -----------------------------------------------------------------------
    # ANALYSIS 5: Sub-clustering within linear-chain family
    # -----------------------------------------------------------------------
    subcluster_results = subclustering_analysis(schemas, primary_ged)

    # -----------------------------------------------------------------------
    # ANALYSIS 5: Summary comparison table
    # -----------------------------------------------------------------------
    print("\n\n  SENSITIVITY COMPARISON TABLE")
    _sep("=")
    print(f"  {'Scenario':<35} {'Chains':>6} {'Branch':>6} {'Intra-C':>8} {'Intra-B':>8} {'Inter':>8} {'Sep':>8}")
    _sep()
    rows = [
        ("Primary (22 schemas)", primary_stats),
        ("Alt 1: Ishraq branching", alt1_stats),
        ("Alt 2: Zurvanite Bundahishn", alt2_stats),
        ("Alt 3: Popol Vuh no Xibalba", alt3_stats),
    ]
    for label, s in rows:
        print(f"  {label:<35} {s['n_chains']:>6} {s['n_branches']:>6} {s['intra_chain']:>8.4f} {s['intra_branch']:>8.4f} {s['inter_family']:>8.4f} {s['separation_ratio']:>7.4f}x")
    _sep("=")

    # -----------------------------------------------------------------------
    # WP 1.5 ANALYSIS 5: Edge-weighted GED sensitivity (4 cost matrices)
    # -----------------------------------------------------------------------
    weighted_ged_results = weighted_ged_sensitivity(schemas)

    # -----------------------------------------------------------------------
    # WP 1.5 ANALYSIS 6: Direction testing for branching trees
    # -----------------------------------------------------------------------
    direction_results = direction_test_branching_trees(schemas)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "primary": primary_stats,
        "primary_families": primary_families,
        "alt1_ishraq_branching": alt1_stats,
        "alt1_families": alt1_families,
        "alt1_ishraq_samkhya_ged": ishraq_samkhya,
        "alt2_zurvanite": alt2_stats,
        "alt2_families": alt2_families,
        "alt2_bundahishn_ishraq_ged": bund_ishraq,
        "alt3_popol_vuh_no_xibalba": alt3_stats,
        "alt3_popol_vuh_classification": popol_class,
        "alt3_popol_vuh_node_count": popol_nodes,
        "alt3_popol_vuh_max_branching": popol_mb,
        "alt4_gospel_direction_sensitive": {
            "classification": gospel_class,
            "ged_values_changed": n_changed,
            "ged_values_total": n_total,
            "direction_agnostic_validated": n_changed == 0,
        },
        "subclustering": subcluster_results,
        "wp15_weighted_ged_sensitivity": weighted_ged_results,
        "wp15_direction_branching_trees": direction_results,
    }

    out_path = os.path.join(OUTPUTS_DIR, "sensitivity_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    _sep("=")
    print()


if __name__ == "__main__":
    main()

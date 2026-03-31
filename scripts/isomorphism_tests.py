"""
isomorphism_tests.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Step 5: Structural isomorphism and similarity tests across all 15 tradition pairs.

Tests performed
---------------
A  Exact structural isomorphism (VF2)
     Unlabeled: are two schemas topologically identical ignoring roles?
     Role-labeled: are they isomorphic when functional_role must also match?

B  Graph Edit Distance (GED)
     Structural GED (unlabeled): minimum edits to transform G1 into G2.
     Role-labeled GED: edits weighted by role substitution cost.
     Normalized GED: GED / (|V1|+|E1|+|V2|+|E2|) * 2  in [0,1].

C  Weisfeiler-Leman (WL) similarity kernel
     Structural WL: initial labels = out-degree.
     Role WL: initial labels = functional_role.
     3 iterations; cosine similarity of WL feature vectors.

D  Role sequence Levenshtein distance
     Flatten role_sequence_by_level into an ordered tuple list.
     Compute pairwise sequence edit distance; normalize by max length.

E  Subgraph isomorphism (VF2)
     For each ordered pair (A,B): is A a structural subgraph of B?

F  Topology family clustering
     Assign each tradition to a family based on GED neighbourhood.
     Report intra-family vs inter-family mean GED.

Outputs
-------
  outputs/similarity_matrix/structural_ged.json
  outputs/similarity_matrix/role_ged.json
  outputs/similarity_matrix/wl_structural.json
  outputs/similarity_matrix/wl_roles.json
  outputs/similarity_matrix/role_levenshtein.json
  outputs/similarity_matrix/subgraph_membership.json
  outputs/similarity_matrix/isomorphism_report.json

Usage:
    python isomorphism_tests.py
"""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import networkx as nx
from networkx.algorithms import isomorphism as nx_iso

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from encode_schemas import load_all_schemas, SCHEMAS_DIR
from compute_invariants import compute_invariants

SIM_DIR = os.path.join(_HERE, "..", "outputs", "similarity_matrix")

WL_ITERATIONS = 3
GED_TIMEOUT = 10  # seconds per pair; 9-node trees resolve well within this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairs(traditions):
    return list(combinations(traditions, 2))


def _matrix_key(a, b):
    return f"{a} vs {b}"


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_graphs():
    schemas = load_all_schemas(SCHEMAS_DIR)
    graphs = {t: G for t, (G, _) in schemas.items()}
    invs = {t: compute_invariants(t, G) for t, G in graphs.items()}
    return graphs, invs


# ---------------------------------------------------------------------------
# Test A: Exact structural isomorphism
# ---------------------------------------------------------------------------

def test_a_exact_isomorphism(graphs: dict) -> dict:
    """
    For every pair: check exact structural isomorphism (unlabeled)
    and role-labeled isomorphism (functional_role must match).
    """
    traditions = sorted(graphs.keys())
    results = {}

    for a, b in _pairs(traditions):
        Ga, Gb = graphs[a], graphs[b]

        struct_iso = nx.is_isomorphic(Ga, Gb)

        def role_match(n1_attrs, n2_attrs):
            return n1_attrs.get("functional_role") == n2_attrs.get("functional_role")

        role_iso = nx.is_isomorphic(Ga, Gb, node_match=role_match)

        results[_matrix_key(a, b)] = {
            "structural_isomorphic": struct_iso,
            "role_labeled_isomorphic": role_iso,
        }

    return results


# ---------------------------------------------------------------------------
# Test B: Graph Edit Distance
# ---------------------------------------------------------------------------

def _role_subst_cost(attrs1, attrs2):
    """Node substitution cost: 0 if same functional_role, 1 otherwise."""
    r1 = attrs1.get("functional_role", "")
    r2 = attrs2.get("functional_role", "")
    return 0.0 if r1 == r2 else 1.0


def _edge_subst_cost(attrs1, attrs2):
    """Edge substitution cost: 0 if same relationship type, 1 otherwise."""
    e1 = attrs1.get("relationship", "")
    e2 = attrs2.get("relationship", "")
    return 0.0 if e1 == e2 else 1.0


# ---------------------------------------------------------------------------
# Edge-weighted GED: semantic edge-type similarity costs (WP 1.4)
# ---------------------------------------------------------------------------

# Cost matrix for edge-type substitution in weighted GED.
# Same type = 0.0, related pair = 0.5, unrelated = 1.0 (default), opposites = 1.5
_EDGE_TYPE_COSTS = {
    # Related pairs (processes that are conceptually close)
    ("emanation", "creation"): 0.5,
    ("creation", "emanation"): 0.5,
    ("fragmentation", "contraction"): 0.5,
    ("contraction", "fragmentation"): 0.5,
    ("reflection", "succession"): 0.5,
    ("succession", "reflection"): 0.5,
    # Opposite pairs (processes that are conceptually opposed)
    ("creation", "fragmentation"): 1.5,
    ("fragmentation", "creation"): 1.5,
    ("emanation", "contraction"): 1.5,
    ("contraction", "emanation"): 1.5,
}


def _edge_subst_cost_weighted(attrs1, attrs2):
    """Edge substitution cost using semantic edge-type similarity.

    Same type = 0.0, related pair = 0.5, unrelated = 1.0, opposites = 1.5.
    Related: emanation/creation, fragmentation/contraction, reflection/succession.
    Opposites: creation/fragmentation, emanation/contraction.
    """
    r1 = attrs1.get("relationship", "")
    r2 = attrs2.get("relationship", "")
    if r1 == r2:
        return 0.0
    return _EDGE_TYPE_COSTS.get((r1, r2), 1.0)


def test_b_ged(graphs: dict) -> dict:
    """
    Compute structural and role-labeled GED for all 15 pairs.
    Normalized GED uses total graph size as denominator.
    """
    traditions = sorted(graphs.keys())
    results = {}

    for a, b in _pairs(traditions):
        Ga, Gb = graphs[a], graphs[b]
        denom = (Ga.number_of_nodes() + Ga.number_of_edges()
                 + Gb.number_of_nodes() + Gb.number_of_edges())

        # Structural GED (topology only)
        struct_ged = nx.graph_edit_distance(Ga, Gb, timeout=GED_TIMEOUT)

        # Role-labeled GED
        role_ged = nx.graph_edit_distance(
            Ga, Gb,
            node_subst_cost=_role_subst_cost,
            edge_subst_cost=_edge_subst_cost,
            timeout=GED_TIMEOUT,
        )

        results[_matrix_key(a, b)] = {
            "structural_ged": struct_ged,
            "role_ged": round(role_ged, 4) if role_ged is not None else None,
            "normalized_structural_ged": round(struct_ged * 2 / denom, 4) if struct_ged is not None else None,
            "normalized_role_ged": round(role_ged * 2 / denom, 4) if role_ged is not None else None,
        }

    return results


def test_b_weighted_ged(graphs: dict) -> dict:
    """
    Compute edge-weighted GED for all pairs (WP 1.4 extension).

    Uses semantic edge-type similarity costs to distinguish traditions that
    are structurally isomorphic (GED=0) but use different edge types.
    E.g., Derveni (succession) vs. Plotinus (emanation) should have weighted GED > 0.
    """
    traditions = sorted(graphs.keys())
    results = {}

    for a, b in _pairs(traditions):
        Ga, Gb = graphs[a], graphs[b]
        denom = (Ga.number_of_nodes() + Ga.number_of_edges()
                 + Gb.number_of_nodes() + Gb.number_of_edges())

        ged = nx.graph_edit_distance(
            Ga, Gb,
            node_subst_cost=_role_subst_cost,
            edge_subst_cost=_edge_subst_cost_weighted,
            timeout=GED_TIMEOUT,
        )

        results[_matrix_key(a, b)] = {
            "weighted_ged": round(ged, 4) if ged is not None else None,
            "normalized_weighted_ged": (
                round(ged * 2 / denom, 4)
                if ged is not None and denom > 0 else None
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Test C: Weisfeiler-Leman similarity kernel
# ---------------------------------------------------------------------------

def _wl_feature_vector(G: nx.DiGraph, initial_labels: dict, n_iter: int = WL_ITERATIONS) -> dict:
    """
    Compute the WL feature vector (histogram of compressed labels across iterations).
    initial_labels: {node_id: label_string}
    Returns: {label_key: count} summed over all iterations.
    """
    labels = dict(initial_labels)
    feature_vec = defaultdict(int)

    # Iteration 0: initial labels
    for lbl in labels.values():
        feature_vec[f"iter0_{lbl}"] += 1

    for it in range(1, n_iter + 1):
        new_labels = {}
        for node in G.nodes:
            # Aggregate: current label + sorted successor labels
            successors = sorted(labels.get(s, "") for s in G.successors(node))
            predecessors = sorted(labels.get(p, "") for p in G.predecessors(node))
            raw = f"{labels[node]}|{','.join(predecessors)}|{','.join(successors)}"
            # Compress to short hash for efficiency
            compressed = hashlib.md5(raw.encode()).hexdigest()[:8]
            new_labels[node] = compressed
            feature_vec[f"iter{it}_{compressed}"] += 1
        labels = new_labels

    return dict(feature_vec)


def _cosine_similarity(v1: dict, v2: dict) -> float:
    """Cosine similarity between two sparse feature vectors (dicts)."""
    keys = set(v1.keys()) | set(v2.keys())
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    norm1 = sum(x ** 2 for x in v1.values()) ** 0.5
    norm2 = sum(x ** 2 for x in v2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(dot / (norm1 * norm2), 6)


def test_c_wl_similarity(graphs: dict) -> tuple[dict, dict]:
    """
    Compute WL similarity for all 15 pairs.
    Returns (structural_results, role_results).
    """
    traditions = sorted(graphs.keys())

    # Precompute feature vectors for each tradition
    struct_fvecs = {}
    role_fvecs = {}

    for t, G in graphs.items():
        # Structural: initial label = out-degree (topology fingerprint)
        struct_labels = {n: str(G.out_degree(n)) for n in G.nodes}
        struct_fvecs[t] = _wl_feature_vector(G, struct_labels)

        # Role: initial label = functional_role
        role_labels = {n: G.nodes[n].get("functional_role", "unknown") for n in G.nodes}
        role_fvecs[t] = _wl_feature_vector(G, role_labels)

    struct_results = {}
    role_results = {}

    for a, b in _pairs(traditions):
        key = _matrix_key(a, b)
        struct_results[key] = {
            "wl_structural_similarity": _cosine_similarity(struct_fvecs[a], struct_fvecs[b])
        }
        role_results[key] = {
            "wl_role_similarity": _cosine_similarity(role_fvecs[a], role_fvecs[b])
        }

    return struct_results, role_results


# ---------------------------------------------------------------------------
# Test D: Role sequence Levenshtein distance
# ---------------------------------------------------------------------------

def _seq_edit_distance(seq1: list, seq2: list) -> int:
    """
    Standard dynamic-programming edit distance on lists.
    Each element is a frozenset of role strings at that level.
    Cost: insert/delete = 1, substitute = 0 if equal else 1.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[m][n]


def _role_sequence(inv: dict) -> list:
    """
    Flatten role_sequence_by_level into an ordered list of frozensets.
    Each element = frozenset of roles at that depth level.
    """
    seq = inv.get("role_sequence_by_level", {})
    return [frozenset(seq[str(k)]) for k in sorted(int(k) for k in seq.keys())]


def test_d_role_levenshtein(invs: dict) -> dict:
    """Pairwise role sequence edit distance for all 15 pairs."""
    traditions = sorted(invs.keys())
    results = {}

    for a, b in _pairs(traditions):
        seq_a = _role_sequence(invs[a])
        seq_b = _role_sequence(invs[b])
        dist = _seq_edit_distance(seq_a, seq_b)
        norm_dist = round(dist / max(len(seq_a), len(seq_b)), 4) if max(len(seq_a), len(seq_b)) > 0 else 0.0
        results[_matrix_key(a, b)] = {
            "role_seq_edit_distance": dist,
            "normalized_role_seq_distance": norm_dist,
            "seq_lengths": [len(seq_a), len(seq_b)],
        }

    return results


# ---------------------------------------------------------------------------
# Test E: Subgraph isomorphism
# ---------------------------------------------------------------------------

def test_e_subgraph(graphs: dict) -> dict:
    """
    For each ordered pair (A, B): is A a structural subgraph of B?
    Returns results for all 30 ordered pairs (both directions).
    """
    traditions = sorted(graphs.keys())
    results = {}

    for a in traditions:
        for b in traditions:
            if a == b:
                continue
            Ga, Gb = graphs[a], graphs[b]
            # Only worth checking if |Va| <= |Vb|
            if Ga.number_of_nodes() > Gb.number_of_nodes():
                results[f"{a} in {b}"] = False
                continue
            gm = nx_iso.DiGraphMatcher(Gb, Ga)
            results[f"{a} in {b}"] = gm.subgraph_is_isomorphic()

    # Filter to only the True cases for readability
    confirmed = {k: v for k, v in results.items() if v}
    return {
        "all_pairs": results,
        "confirmed_subgraph_relations": confirmed,
        "count_confirmed": len(confirmed),
    }


# ---------------------------------------------------------------------------
# Test F: Topology family clustering
# ---------------------------------------------------------------------------

def test_f_topology_families(ged_results: dict, graphs: dict, invs: dict) -> dict:
    """
    Assign traditions to topology families based on structural GED.
    Families defined by shared shape characteristics:
      - Linear chains (max_branching=1)
      - Branching trees (max_branching>1)
    Report intra-family and inter-family mean GED.
    """
    traditions = sorted(graphs.keys())

    families = {
        "linear_chain": [t for t in traditions if invs[t]["is_linear_chain"]],
        "branching_tree": [t for t in traditions if not invs[t]["is_linear_chain"]],
    }

    def mean_ged_within(family_members):
        pairs = list(combinations(family_members, 2))
        if not pairs:
            return None
        vals = [ged_results[_matrix_key(a, b)]["structural_ged"] for a, b in pairs]
        return round(float(np.mean(vals)), 4)

    def mean_ged_between(family_a, family_b):
        vals = []
        for a in family_a:
            for b in family_b:
                key = _matrix_key(a, b) if a < b else _matrix_key(b, a)
                vals.append(ged_results[key]["structural_ged"])
        return round(float(np.mean(vals)), 4) if vals else None

    intra_chain = mean_ged_within(families["linear_chain"])
    intra_tree = mean_ged_within(families["branching_tree"])
    inter = mean_ged_between(families["linear_chain"], families["branching_tree"])

    # Build full GED matrix for reference
    ged_matrix = {}
    for t in traditions:
        ged_matrix[t] = {}
        for t2 in traditions:
            if t == t2:
                ged_matrix[t][t2] = 0
            else:
                key = _matrix_key(t, t2) if t < t2 else _matrix_key(t2, t)
                ged_matrix[t][t2] = ged_results[key]["structural_ged"]

    return {
        "families": families,
        "family_sizes": {f: len(m) for f, m in families.items()},
        "intra_family_mean_ged": {
            "linear_chain": intra_chain,
            "branching_tree": intra_tree,
        },
        "inter_family_mean_ged": inter,
        "family_separation_ratio": (
            round(inter / intra_chain, 4)
            if inter and intra_chain and intra_chain > 0 else None
        ),
        "ged_matrix": ged_matrix,
        "interpretation": (
            f"Linear chain family ({len(families['linear_chain'])} members) has mean intra-family GED = {intra_chain}. "
            f"Branching tree family ({len(families['branching_tree'])} members) mean = {intra_tree}. "
            f"Inter-family mean GED = {inter}. "
            + (f"Separation ratio = {round(inter/intra_chain, 2)}x." if inter and intra_chain else "")
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate runner and report
# ---------------------------------------------------------------------------

def run_all(schemas_dir: str = SCHEMAS_DIR) -> dict:
    os.makedirs(SIM_DIR, exist_ok=True)
    graphs, invs = _load_graphs()

    results = {}
    results["test_a_isomorphism"] = test_a_exact_isomorphism(graphs)
    results["test_b_ged"] = test_b_ged(graphs)
    results["test_b_weighted_ged"] = test_b_weighted_ged(graphs)
    wl_struct, wl_roles = test_c_wl_similarity(graphs)
    results["test_c_wl_structural"] = wl_struct
    results["test_c_wl_roles"] = wl_roles
    results["test_d_role_levenshtein"] = test_d_role_levenshtein(invs)
    results["test_e_subgraph"] = test_e_subgraph(graphs)
    results["test_f_families"] = test_f_topology_families(results["test_b_ged"], graphs, invs)

    # Save individual matrices
    _save(os.path.join(SIM_DIR, "structural_ged.json"), results["test_b_ged"])
    _save(os.path.join(SIM_DIR, "role_ged.json"),
          {k: {"role_ged": v["role_ged"], "normalized_role_ged": v["normalized_role_ged"]}
           for k, v in results["test_b_ged"].items()})
    _save(os.path.join(SIM_DIR, "weighted_ged.json"), results["test_b_weighted_ged"])
    _save(os.path.join(SIM_DIR, "wl_structural.json"), results["test_c_wl_structural"])
    _save(os.path.join(SIM_DIR, "wl_roles.json"), results["test_c_wl_roles"])
    _save(os.path.join(SIM_DIR, "role_levenshtein.json"), results["test_d_role_levenshtein"])
    _save(os.path.join(SIM_DIR, "subgraph_membership.json"), results["test_e_subgraph"])
    _save(os.path.join(SIM_DIR, "isomorphism_report.json"), results)

    return results, graphs, invs


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def _sep(char="-", width=72):
    print(char * width)


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "#" * filled + "." * (width - filled)


def print_report(results: dict, graphs: dict, invs: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    traditions = sorted(graphs.keys())

    print()
    _sep("=")
    print("  WP 1.1 -- isomorphism_tests.py")
    print("  Structural Similarity Report -- All 15 Tradition Pairs")
    _sep("=")

    # --- Test A ---
    print("\n  TEST A: Exact Structural Isomorphism (VF2)")
    _sep()
    ta = results["test_a_isomorphism"]
    any_iso = any(v["structural_isomorphic"] for v in ta.values())
    print(f"  Unlabeled isomorphic pairs:    {'NONE' if not any_iso else ''}")
    any_role_iso = any(v["role_labeled_isomorphic"] for v in ta.values())
    print(f"  Role-labeled isomorphic pairs: {'NONE' if not any_role_iso else ''}")
    if not any_iso and not any_role_iso:
        print("  All 6 schemas are structurally distinct -- confirmed by VF2.")
    _sep()

    # --- Test B: GED ranked table ---
    print("\n  TEST B: Graph Edit Distance (structural / role-labeled / normalized)")
    _sep()
    print(f"  {'Pair':<55}  sGED  rGED  nGED")
    _sep()
    tb = results["test_b_ged"]
    for pair, vals in sorted(tb.items(), key=lambda x: x[1]["structural_ged"]):
        sged = vals["structural_ged"]
        rged = vals["role_ged"]
        nged = vals["normalized_structural_ged"]
        bar = _bar(1 - nged)
        print(f"  {pair:<55}  {sged:>4}  {rged:>4}  {nged:.3f}  {bar}")
    _sep()
    print("  sGED=structural GED  rGED=role-labeled GED  "
          "nGED=normalized (lower=more similar)")

    # --- Test B (weighted): Edge-weighted GED ---
    print("\n  TEST B (WEIGHTED): Edge-Weighted GED (semantic edge-type costs)")
    _sep()
    print(f"  {'Pair':<55}  wGED  nwGED")
    _sep()
    tbw = results.get("test_b_weighted_ged", {})
    # Show only pairs where weighted GED differs from structural GED
    divergent = []
    for pair in sorted(tbw.keys()):
        wged = tbw[pair]["weighted_ged"]
        sged = tb.get(pair, {}).get("structural_ged")
        if wged is not None and sged is not None and abs(wged - sged) > 0.01:
            divergent.append(pair)
    if divergent:
        print("  Pairs where weighted GED differs from structural GED:")
        for pair in sorted(divergent, key=lambda p: tbw[p]["weighted_ged"]):
            wged = tbw[pair]["weighted_ged"]
            nwged = tbw[pair]["normalized_weighted_ged"]
            sged = tb[pair]["structural_ged"]
            print(f"  {pair:<55}  {wged:>4}  {nwged:.3f}  (sGED={sged})")
    else:
        print("  No divergence between weighted and structural GED.")
    _sep()

    # --- Test C: WL similarity ranked ---
    print("\n  TEST C: Weisfeiler-Leman Similarity (3 iterations)")
    _sep()
    print(f"  {'Pair':<55}  WL-struct  WL-roles")
    _sep()
    tc_s = results["test_c_wl_structural"]
    tc_r = results["test_c_wl_roles"]
    combined = {
        k: (tc_s[k]["wl_structural_similarity"], tc_r[k]["wl_role_similarity"])
        for k in tc_s
    }
    for pair, (ws, wr) in sorted(combined.items(), key=lambda x: -x[1][1]):
        print(f"  {pair:<55}  {ws:.4f}    {wr:.4f}  {_bar(wr)}")
    _sep()

    # --- Test D: Role Levenshtein ---
    print("\n  TEST D: Role Sequence Edit Distance (normalized)")
    _sep()
    print(f"  {'Pair':<55}  Raw  Norm  Similarity")
    _sep()
    td = results["test_d_role_levenshtein"]
    for pair, vals in sorted(td.items(), key=lambda x: x[1]["normalized_role_seq_distance"]):
        raw = vals["role_seq_edit_distance"]
        norm = vals["normalized_role_seq_distance"]
        sim = 1 - norm
        print(f"  {pair:<55}  {raw:>3}  {norm:.3f}  {_bar(sim)}")
    _sep()

    # --- Test E: Subgraph ---
    print("\n  TEST E: Subgraph Isomorphism")
    _sep()
    te = results["test_e_subgraph"]
    confirmed = te["confirmed_subgraph_relations"]
    if confirmed:
        for rel in confirmed:
            print(f"  CONFIRMED: {rel}")
    else:
        print("  No subgraph isomorphism found -- no tradition is a structural")
        print("  subgraph of any other. All schemas are topologically independent.")
    _sep()

    # --- Test F: Families ---
    print("\n  TEST F: Topology Family Clustering")
    _sep()
    tf = results["test_f_families"]
    for family, members in tf["families"].items():
        print(f"  {family}: {', '.join(members)}")
    print()
    print(f"  Intra-family mean GED (linear chains):  "
          f"{tf['intra_family_mean_ged']['linear_chain']}")
    print(f"  Intra-family mean GED (branching trees): "
          f"{tf['intra_family_mean_ged']['branching_tree']}")
    print(f"  Inter-family mean GED:                   {tf['inter_family_mean_ged']}")
    print(f"  Separation ratio:                        {tf['family_separation_ratio']}x")
    print(f"\n  {tf['interpretation']}")
    _sep()

    # --- GED full matrix ---
    print("\n  FULL STRUCTURAL GED MATRIX")
    _sep()
    ged_m = tf["ged_matrix"]
    short = {t: t[:8] for t in traditions}
    header = "  " + " " * 10 + "  ".join(f"{short[t]:>8}" for t in traditions)
    print(header)
    _sep()
    for t in traditions:
        row = f"  {short[t]:<10}"
        for t2 in traditions:
            val = ged_m[t][t2]
            row += f"  {val:>8}"
        print(row)
    _sep()

    # --- Summary ---
    print("\n  KEY FINDINGS SUMMARY")
    _sep("=")

    # Most similar pair by each metric
    min_ged_pair = min(tb.items(), key=lambda x: x[1]["structural_ged"])
    max_ged_pair = max(tb.items(), key=lambda x: x[1]["structural_ged"])
    max_wl_pair = max(combined.items(), key=lambda x: x[1][1])
    min_lev_pair = min(td.items(), key=lambda x: x[1]["normalized_role_seq_distance"])

    print(f"  Most structurally similar  (min GED):  {min_ged_pair[0]}  "
          f"(GED={min_ged_pair[1]['structural_ged']})")
    print(f"  Most structurally distant  (max GED):  {max_ged_pair[0]}  "
          f"(GED={max_ged_pair[1]['structural_ged']})")
    print(f"  Highest WL role similarity:            {max_wl_pair[0]}  "
          f"(WL={max_wl_pair[1][1]:.4f})")
    print(f"  Most similar role sequences:           {min_lev_pair[0]}  "
          f"(norm-dist={min_lev_pair[1]['normalized_role_seq_distance']})")
    print(f"  Topology families: {len(tf['families'])} "
          f"(separation ratio = {tf['family_separation_ratio']}x)")
    print(f"  Subgraph relations confirmed: {te['count_confirmed']}")
    _sep("=")
    print()


def main():
    t0 = time.time()
    print("\n  Running isomorphism and similarity tests...")

    results, graphs, invs = run_all()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s\n")
    print_report(results, graphs, invs)
    print(f"  Outputs saved to: {os.path.abspath(SIM_DIR)}\n")


if __name__ == "__main__":
    main()

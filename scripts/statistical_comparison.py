"""
statistical_comparison.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Step 4: Formal statistical tests comparing real schemas against null models.

Tests performed
---------------
Test 1  Per-tradition extremity
        For each tradition x metric: z-score, one-sided permutation p-value,
        Cohen's d against the stratified null (same node count).

Test 2  Cross-corpus depth variance (convergence test)
        H0: variance of depth across 6 real schemas equals what you'd get
            by drawing 6 random trees of the same sizes.
        H1: real schemas are MORE similar in depth than chance.
        Method: 10,000 permutation draws from stratified controls.

Test 3  Linear chain frequency (exact binomial)
        H0: each schema has its baseline p(chain) chance of being linear.
        H1: corpus has significantly more chains than baseline predicts.
        Method: exact binomial probability using per-tradition chain rates.

Test 4  Corpus depth distribution vs. pooled null (Mann-Whitney U)
        H0: real schema depths are drawn from the same distribution as
            random DAG trees.
        H1: real schemas are significantly deeper.

Test 5  Pairwise depth similarity (corpus-level convergence)
        For all 15 tradition pairs: compute |depth_i - depth_j|.
        Null: 10,000 random draws of 6 trees (one per tradition from
        stratified controls), compute mean pairwise depth difference.
        p-value: proportion of null runs where mean diff <= real mean diff.

Test 6  Role sequence similarity matrix
        For all 15 tradition pairs: proportion of depth levels where the
        sorted functional role list matches exactly.
        No null comparison (controls have no roles) -- reports raw matrix.

Outputs
-------
  outputs/invariants/statistical_results.json   -- full results
  outputs/invariants/statistical_report.txt     -- human-readable summary

Usage:
    python statistical_comparison.py
"""

import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy import stats as scipy_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from encode_schemas import load_all_schemas, SCHEMAS_DIR
from compute_invariants import compute_invariants

CONTROLS_DIR = os.path.join(_HERE, "..", "outputs", "invariants", "controls")
OUTPUTS_DIR = os.path.join(_HERE, "..", "outputs", "invariants")

N_PERMUTATIONS = 10_000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_controls() -> dict:
    """Return {tradition: [control_dict, ...]} plus 'pooled' key."""
    controls = {"pooled": _load_json(os.path.join(CONTROLS_DIR, "pooled_controls.json"))}
    for fname in os.listdir(CONTROLS_DIR):
        if fname.startswith("stratified_") and fname.endswith(".json"):
            tradition = fname[len("stratified_"):-len(".json")]
            controls[tradition] = _load_json(os.path.join(CONTROLS_DIR, fname))
    return controls


# ---------------------------------------------------------------------------
# Test 1: Per-tradition extremity
# ---------------------------------------------------------------------------

METRICS_T1 = ["depth", "width", "max_branching", "mean_branching", "leaf_count"]


def test1_per_tradition_extremity(
    real_inv: dict, controls: dict
) -> dict:
    """
    For each tradition x metric, compute:
      - null_mean, null_std
      - z_score = (real - null_mean) / null_std
      - one_sided_p (permutation: proportion of controls >= real value)
      - cohen_d = z_score (identical for one-sample standardization)
      - significant at alpha=0.05
    """
    results = {}

    for tradition, inv in real_inv.items():
        strat = controls.get(tradition, [])
        if not strat:
            continue

        tradition_results = {}
        for metric in METRICS_T1:
            real_val = inv.get(metric)
            if real_val is None:
                continue

            null_vals = [c[metric] for c in strat if metric in c]
            if not null_vals:
                continue

            null_arr = np.array(null_vals, dtype=float)
            null_mean = float(np.mean(null_arr))
            null_std = float(np.std(null_arr, ddof=1))

            z = (real_val - null_mean) / null_std if null_std > 0 else 0.0

            # One-sided p: proportion of controls >= real (testing for unusually HIGH values)
            # For width and branching, we might want to test for LOW values -- handled below
            if metric in ("width", "mean_branching", "max_branching", "leaf_count"):
                # LOW values = more chain-like = the surprising direction
                p_one_sided = float(np.mean(null_arr <= real_val))
                direction = "low"
            else:
                # HIGH values (depth) = deeper = the surprising direction
                p_one_sided = float(np.mean(null_arr >= real_val))
                direction = "high"

            tradition_results[metric] = {
                "real_value": real_val,
                "null_mean": round(null_mean, 4),
                "null_std": round(null_std, 4),
                "z_score": round(z, 4),
                "cohen_d": round(abs(z), 4),
                "p_one_sided": round(p_one_sided, 4),
                "direction_tested": direction,
                "significant": p_one_sided < ALPHA,
            }

        results[tradition] = tradition_results

    return results


# ---------------------------------------------------------------------------
# Test 2: Cross-corpus depth variance (convergence test)
# ---------------------------------------------------------------------------

def test2_depth_variance(real_inv: dict, controls: dict, n_perm: int = N_PERMUTATIONS) -> dict:
    """
    H0: variance of depths across 6 real schemas = variance from 6 random same-sized trees.
    H1: real schemas have LOWER depth variance (more convergent) than chance.
    """
    traditions = sorted(real_inv.keys())
    real_depths = [real_inv[t]["depth"] for t in traditions]
    real_variance = float(np.var(real_depths, ddof=1))

    rng = random.Random(99)
    null_variances = []
    for _ in range(n_perm):
        sample_depths = [
            rng.choice(controls[t])["depth"]
            for t in traditions
            if t in controls
        ]
        null_variances.append(float(np.var(sample_depths, ddof=1)))

    null_arr = np.array(null_variances)
    # p = proportion of null runs with variance <= real variance (testing for convergence)
    p_value = float(np.mean(null_arr <= real_variance))
    # For low variance = convergence: p = proportion <= real is NOT what we want.
    # We want: proportion of nulls as convergent or more convergent.
    p_convergence = float(np.mean(null_arr <= real_variance))

    return {
        "real_depths": {t: real_inv[t]["depth"] for t in traditions},
        "real_variance": round(real_variance, 4),
        "real_std": round(float(np.std(real_depths, ddof=1)), 4),
        "null_variance_mean": round(float(np.mean(null_arr)), 4),
        "null_variance_std": round(float(np.std(null_arr, ddof=1)), 4),
        "null_variance_p05": round(float(np.percentile(null_arr, 5)), 4),
        "null_variance_p50": round(float(np.percentile(null_arr, 50)), 4),
        "null_variance_p95": round(float(np.percentile(null_arr, 95)), 4),
        "p_convergence": round(p_convergence, 4),
        "significant_convergence": p_convergence < ALPHA,
        "n_permutations": n_perm,
        "interpretation": (
            "Real schemas have LOWER depth variance than expected by chance "
            "(evidence for cross-traditional convergence)"
            if p_convergence < ALPHA else
            "Real schemas do not show significantly lower depth variance than chance"
        ),
    }


# ---------------------------------------------------------------------------
# Test 3: Linear chain frequency (exact binomial)
# ---------------------------------------------------------------------------

def test3_chain_frequency(real_inv: dict, controls: dict) -> dict:
    """
    Observed: k chains among 6 traditions.
    Null: each tradition i has p_i = empirical chain rate in its stratified control.
    Method: exact Monte Carlo binomial (accounts for different p_i per tradition).
    """
    traditions = sorted(real_inv.keys())
    observed_chains = sum(1 for t in traditions if real_inv[t]["is_linear_chain"])
    n = len(traditions)

    # Per-tradition baseline chain rates from stratified controls
    baseline_rates = {}
    for t in traditions:
        strat = controls.get(t, [])
        if strat:
            baseline_rates[t] = sum(1 for c in strat if c.get("is_linear_chain", False)) / len(strat)
        else:
            baseline_rates[t] = 0.053  # pooled fallback

    # Monte Carlo: draw 6 Bernoulli trials (one per tradition) 100,000 times
    rng = random.Random(7)
    n_mc = 100_000
    null_chain_counts = []
    tradition_rates = [baseline_rates[t] for t in traditions]
    for _ in range(n_mc):
        count = sum(1 for p in tradition_rates if rng.random() < p)
        null_chain_counts.append(count)

    null_arr = np.array(null_chain_counts)
    # p = P(X >= observed) under null
    p_value = float(np.mean(null_arr >= observed_chains))

    return {
        "observed_chains": observed_chains,
        "n_traditions": n,
        "chain_traditions": [t for t in traditions if real_inv[t]["is_linear_chain"]],
        "non_chain_traditions": [t for t in traditions if not real_inv[t]["is_linear_chain"]],
        "observed_frequency": round(observed_chains / n, 4),
        "baseline_rates_per_tradition": {t: round(baseline_rates[t], 4) for t in traditions},
        "expected_chains_mean": round(float(np.mean(null_arr)), 4),
        "p_value": round(p_value, 6),
        "significant": p_value < ALPHA,
        "n_monte_carlo": n_mc,
        "interpretation": (
            f"Observed {observed_chains}/{n} linear chains. "
            f"p = {p_value:.4f}. "
            + (
                "SIGNIFICANT: corpus contains significantly more linear chains "
                "than baseline predicts."
                if p_value < ALPHA else
                "Not significant at alpha=0.05."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Test 4: Corpus depth distribution vs. pooled null (Mann-Whitney U)
# ---------------------------------------------------------------------------

def test4_mann_whitney(real_inv: dict, controls: dict) -> dict:
    """
    One-sided Mann-Whitney U test.
    H0: real schema depths come from the same distribution as pooled null depths.
    H1: real schemas are significantly deeper.
    """
    real_depths = [real_inv[t]["depth"] for t in real_inv]
    pooled_depths = [c["depth"] for c in controls["pooled"]]

    u_stat, p_two_sided = scipy_stats.mannwhitneyu(
        real_depths, pooled_depths, alternative="greater"
    )

    null_arr = np.array(pooled_depths, dtype=float)
    real_arr = np.array(real_depths, dtype=float)

    return {
        "real_depths": sorted(real_depths),
        "real_mean": round(float(np.mean(real_arr)), 4),
        "real_median": round(float(np.median(real_arr)), 4),
        "null_mean": round(float(np.mean(null_arr)), 4),
        "null_median": round(float(np.median(null_arr)), 4),
        "null_std": round(float(np.std(null_arr, ddof=1)), 4),
        "U_statistic": round(float(u_stat), 4),
        "p_one_sided": round(float(p_two_sided), 6),
        "significant": float(p_two_sided) < ALPHA,
        "interpretation": (
            "Real schemas are significantly deeper than random DAG trees of similar size."
            if float(p_two_sided) < ALPHA else
            "Real schema depths are not significantly different from random trees."
        ),
    }


# ---------------------------------------------------------------------------
# Test 5: Pairwise depth similarity (cross-tradition convergence)
# ---------------------------------------------------------------------------

def test5_pairwise_depth(real_inv: dict, controls: dict, n_perm: int = N_PERMUTATIONS) -> dict:
    """
    For all 15 tradition pairs: compute |depth_i - depth_j|.
    Null: 10,000 random draws of one tree per tradition from stratified controls.
    p-value: proportion of null runs where mean pairwise |diff| <= real mean.
    """
    traditions = sorted(real_inv.keys())
    pairs = [(traditions[i], traditions[j])
             for i in range(len(traditions))
             for j in range(i + 1, len(traditions))]

    real_diffs = {
        f"{a} vs {b}": abs(real_inv[a]["depth"] - real_inv[b]["depth"])
        for a, b in pairs
    }
    real_mean_diff = float(np.mean(list(real_diffs.values())))

    # Null distribution of mean pairwise |diff|
    rng = random.Random(55)
    null_mean_diffs = []
    for _ in range(n_perm):
        sample = {t: rng.choice(controls[t])["depth"] for t in traditions if t in controls}
        diffs = [abs(sample[a] - sample[b]) for a, b in pairs if a in sample and b in sample]
        null_mean_diffs.append(float(np.mean(diffs)))

    null_arr = np.array(null_mean_diffs)
    p_value = float(np.mean(null_arr <= real_mean_diff))

    return {
        "pairwise_depth_diffs": {k: int(v) for k, v in real_diffs.items()},
        "real_mean_pairwise_diff": round(real_mean_diff, 4),
        "null_mean_diff_mean": round(float(np.mean(null_arr)), 4),
        "null_mean_diff_std": round(float(np.std(null_arr, ddof=1)), 4),
        "null_mean_diff_p05": round(float(np.percentile(null_arr, 5)), 4),
        "null_mean_diff_p50": round(float(np.percentile(null_arr, 50)), 4),
        "null_mean_diff_p95": round(float(np.percentile(null_arr, 95)), 4),
        "p_similarity": round(p_value, 4),
        "significant_similarity": p_value < ALPHA,
        "n_permutations": n_perm,
        "interpretation": (
            "Real schemas are NOT significantly more similar in depth than "
            "random trees of the same sizes. Depth similarity could be due to chance."
            if p_value >= ALPHA else
            "Real schemas are significantly more similar in depth than "
            "random trees of the same sizes (cross-traditional convergence)."
        ),
    }


# ---------------------------------------------------------------------------
# Test 6: Role sequence similarity matrix
# ---------------------------------------------------------------------------

def role_sequence_similarity(inv_a: dict, inv_b: dict) -> float:
    """
    Proportion of depth levels (present in BOTH schemas) where the
    sorted functional role list matches exactly.
    Returns 0.0 (no match) to 1.0 (full match).
    """
    seq_a = inv_a.get("role_sequence_by_level", {})
    seq_b = inv_b.get("role_sequence_by_level", {})

    shared_levels = set(seq_a.keys()) & set(seq_b.keys())
    if not shared_levels:
        return 0.0

    matches = sum(
        1 for lv in shared_levels
        if sorted(seq_a[lv]) == sorted(seq_b[lv])
    )
    return round(matches / len(shared_levels), 4)


def test6_role_similarity_matrix(real_inv: dict) -> dict:
    """
    Compute pairwise role sequence similarity for all 15 tradition pairs.
    Also reports: mean similarity, most similar pair, most different pair.
    """
    traditions = sorted(real_inv.keys())
    matrix = {}
    all_scores = []

    for i, t_a in enumerate(traditions):
        for j, t_b in enumerate(traditions):
            if i >= j:
                continue
            score = role_sequence_similarity(real_inv[t_a], real_inv[t_b])
            key = f"{t_a} vs {t_b}"
            matrix[key] = score
            all_scores.append((score, key))

    all_scores.sort(reverse=True)

    return {
        "pairwise_role_similarity": matrix,
        "mean_similarity": round(float(np.mean([s for s, _ in all_scores])), 4),
        "most_similar_pair": all_scores[0][1] if all_scores else None,
        "most_similar_score": all_scores[0][0] if all_scores else None,
        "least_similar_pair": all_scores[-1][1] if all_scores else None,
        "least_similar_score": all_scores[-1][0] if all_scores else None,
        "pairs_above_0.5": [key for score, key in all_scores if score > 0.5],
        "note": (
            "Similarity = proportion of shared depth levels with identical "
            "sorted functional role lists. No null comparison available "
            "(controls have no role assignments)."
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_tests(schemas_dir: str = SCHEMAS_DIR) -> dict:
    schemas = load_all_schemas(schemas_dir)
    real_inv = {t: compute_invariants(t, G) for t, (G, _data) in schemas.items()}
    controls = load_all_controls()

    results = {
        "test_1_per_tradition_extremity": test1_per_tradition_extremity(real_inv, controls),
        "test_2_depth_variance": test2_depth_variance(real_inv, controls),
        "test_3_chain_frequency": test3_chain_frequency(real_inv, controls),
        "test_4_mann_whitney_depth": test4_mann_whitney(real_inv, controls),
        "test_5_pairwise_depth_similarity": test5_pairwise_depth(real_inv, controls),
        "test_6_role_similarity_matrix": test6_role_similarity_matrix(real_inv),
    }
    return results, real_inv


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def _sep(char="-", width=72):
    print(char * width)


def _sig(p: float, alpha: float = ALPHA) -> str:
    if p < 0.001:
        return "*** p<0.001"
    if p < 0.01:
        return f"**  p={p:.4f}"
    if p < alpha:
        return f"*   p={p:.4f}"
    return f"    p={p:.4f} (ns)"


def print_report(results: dict, real_inv: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _sep("=")
    print("  WP 1.1 -- statistical_comparison.py")
    print("  Formal Statistical Results")
    _sep("=")

    # -- Test 1 --
    print("\n  TEST 1: Per-Tradition Extremity vs. Stratified Null")
    print("  (Is each schema unusual for a random tree of its size?)")
    _sep()
    print(f"  {'Tradition':<28}  {'Metric':<18}  {'Real':>5}  "
          f"{'Null mu':>7}  {'Null sd':>7}  {'d':>6}  Sig")
    _sep()
    t1 = results["test_1_per_tradition_extremity"]
    for tradition in sorted(t1.keys()):
        for metric in METRICS_T1:
            r = t1[tradition].get(metric, {})
            if not r:
                continue
            sig_str = "*" if r["significant"] else " "
            print(
                f"  {tradition:<28}  {metric:<18}  {r['real_value']:>5}  "
                f"{r['null_mean']:>7.3f}  {r['null_std']:>7.3f}  "
                f"{r['cohen_d']:>6.3f}  {sig_str} {_sig(r['p_one_sided'])}"
            )
    _sep()
    print("  * = significant at alpha=0.05 (one-sided permutation test)")
    print("  d = Cohen's d (effect size; >0.8 = large)")

    # -- Test 2 --
    print("\n  TEST 2: Cross-Corpus Depth Variance (Convergence Test)")
    print("  (Are the 6 real schemas more similar in depth than chance?)")
    _sep()
    t2 = results["test_2_depth_variance"]
    print(f"  Real depth values: {list(t2['real_depths'].values())}")
    print(f"  Real variance:     {t2['real_variance']}")
    print(f"  Null variance mu:  {t2['null_variance_mean']}  "
          f"(sd={t2['null_variance_std']}, "
          f"p50={t2['null_variance_p50']}, "
          f"p95={t2['null_variance_p95']})")
    print(f"  p (convergence):   {_sig(t2['p_convergence'])}")
    print(f"  Interpretation:    {t2['interpretation']}")
    _sep()

    # -- Test 3 --
    print("\n  TEST 3: Linear Chain Frequency (Exact Monte Carlo Binomial)")
    print("  (Does the corpus have significantly more chains than baseline predicts?)")
    _sep()
    t3 = results["test_3_chain_frequency"]
    print(f"  Observed chains:   {t3['observed_chains']}/{t3['n_traditions']} "
          f"({t3['observed_frequency']*100:.1f}%)")
    print(f"  Chain traditions:  {', '.join(t3['chain_traditions'])}")
    print(f"  Non-chain:         {', '.join(t3['non_chain_traditions'])}")
    print(f"  Baseline rates:    " +
          "  ".join(f"{t}: {r:.3f}" for t, r in t3["baseline_rates_per_tradition"].items()))
    print(f"  Expected (mean):   {t3['expected_chains_mean']:.2f}")
    print(f"  p-value:           {_sig(t3['p_value'])}")
    print(f"  Interpretation:    {t3['interpretation']}")
    _sep()

    # -- Test 4 --
    print("\n  TEST 4: Corpus Depth vs. Pooled Null (Mann-Whitney U, one-sided)")
    print("  (Are real schemas significantly deeper than random DAG trees?)")
    _sep()
    t4 = results["test_4_mann_whitney_depth"]
    print(f"  Real depths:    {t4['real_depths']}  "
          f"(mean={t4['real_mean']}, median={t4['real_median']})")
    print(f"  Null depths:    mean={t4['null_mean']}, median={t4['null_median']}, "
          f"sd={t4['null_std']}")
    print(f"  U statistic:    {t4['U_statistic']}")
    print(f"  p (one-sided):  {_sig(t4['p_one_sided'])}")
    print(f"  Interpretation: {t4['interpretation']}")
    _sep()

    # -- Test 5 --
    print("\n  TEST 5: Pairwise Depth Similarity (Cross-Tradition Convergence)")
    print("  (Are the 6 schemas more similar to each other in depth than chance?)")
    _sep()
    t5 = results["test_5_pairwise_depth_similarity"]
    print(f"  Pairwise |depth_i - depth_j|:")
    for pair, diff in sorted(t5["pairwise_depth_diffs"].items(), key=lambda x: x[1]):
        print(f"    {pair:<55}  diff={diff}")
    print(f"  Real mean pairwise diff:   {t5['real_mean_pairwise_diff']}")
    print(f"  Null mean pairwise diff:   mu={t5['null_mean_diff_mean']}, "
          f"sd={t5['null_mean_diff_std']}, "
          f"p50={t5['null_mean_diff_p50']}, p95={t5['null_mean_diff_p95']}")
    print(f"  p (similarity):            {_sig(t5['p_similarity'])}")
    print(f"  Interpretation:            {t5['interpretation']}")
    _sep()

    # -- Test 6 --
    print("\n  TEST 6: Role Sequence Similarity Matrix")
    print("  (Proportion of shared levels where functional roles match exactly)")
    _sep()
    t6 = results["test_6_role_similarity_matrix"]
    print(f"  {'Pair':<55}  Similarity")
    _sep()
    for pair, score in sorted(
        t6["pairwise_role_similarity"].items(), key=lambda x: -x[1]
    ):
        bar = "#" * int(score * 20)
        print(f"  {pair:<55}  {score:.4f}  {bar}")
    _sep()
    print(f"  Mean similarity:     {t6['mean_similarity']}")
    print(f"  Most similar pair:   {t6['most_similar_pair']} "
          f"(score={t6['most_similar_score']})")
    print(f"  Least similar pair:  {t6['least_similar_pair']} "
          f"(score={t6['least_similar_score']})")
    pairs_above = t6.get("pairs_above_0.5", [])
    print(f"  Pairs with score >0.5:  {len(pairs_above)} of 15")
    if pairs_above:
        for p in pairs_above:
            print(f"    {p}")
    _sep()

    # -- Summary --
    print("\n  SUMMARY OF FINDINGS")
    _sep("=")
    findings = []
    if results["test_2_depth_variance"]["significant_convergence"]:
        findings.append("[SIGNIFICANT] Test 2: Real schemas converge in depth more than chance.")
    else:
        findings.append("[n.s.] Test 2: Depth variance not significantly lower than null.")

    if results["test_3_chain_frequency"]["significant"]:
        findings.append("[SIGNIFICANT] Test 3: Linear chain frequency exceeds baseline expectation.")
    else:
        findings.append("[n.s.] Test 3: Chain frequency not significant.")

    if results["test_4_mann_whitney_depth"]["significant"]:
        findings.append("[SIGNIFICANT] Test 4: Real schemas are significantly deeper than random trees.")
    else:
        findings.append("[n.s.] Test 4: Depth not significantly above null.")

    if results["test_5_pairwise_depth_similarity"]["significant_similarity"]:
        findings.append("[SIGNIFICANT] Test 5: Real schemas are more similar in depth than random pairs.")
    else:
        findings.append("[n.s.] Test 5: Pairwise depth similarity not significant.")

    n_sig_t1 = sum(
        1 for t in results["test_1_per_tradition_extremity"].values()
        for m in t.values() if isinstance(m, dict) and m.get("significant")
    )
    findings.append(f"Test 1: {n_sig_t1} significant tradition x metric combinations.")

    for f in findings:
        print(f"  {f}")
    _sep("=")
    print()


def main():
    t0 = time.time()
    print("\n  Running statistical tests...")

    results, real_inv = run_all_tests()

    # Save JSON results
    results_path = os.path.join(OUTPUTS_DIR, "statistical_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s\n")

    print_report(results, real_inv)

    print(f"  Full results saved to: {os.path.abspath(results_path)}")
    print()


if __name__ == "__main__":
    main()

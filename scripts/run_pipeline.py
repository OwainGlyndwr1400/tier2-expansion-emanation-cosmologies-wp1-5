"""
run_pipeline.py
WP 1.1 -- Emanation Topology Analysis
Pipeline Orchestrator: run all 6 steps in sequence.

Steps
-----
1  encode_schemas.py       Load + validate all 6 tradition schemas
2  compute_invariants.py   Compute topological invariants
3  generate_controls.py    Generate 7,000 randomised null-model DAGs
4  statistical_comparison.py  Run 6 formal statistical tests
5  isomorphism_tests.py    VF2 + WL + GED across all 15 pairs
6  visualize.py            Generate 6 publication-ready figures

Usage
-----
    python run_pipeline.py            # full pipeline
    python run_pipeline.py --from 4   # resume from step 4
    python run_pipeline.py --only 6   # run only step 6
"""

import argparse
import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    {
        "n": 1,
        "name": "encode_schemas",
        "label": "Schema Loading & DAG Contract Validation",
        "module": "encode_schemas",
        "entry": "main",
    },
    {
        "n": 2,
        "name": "compute_invariants",
        "label": "Topological Invariant Computation",
        "module": "compute_invariants",
        "entry": "main",
    },
    {
        "n": 3,
        "name": "generate_controls",
        "label": "Null Model Generation (7,000 random DAGs)",
        "module": "generate_controls",
        "entry": "main",
    },
    {
        "n": 4,
        "name": "statistical_comparison",
        "label": "Statistical Tests (6 tests vs. null distribution)",
        "module": "statistical_comparison",
        "entry": "main",
    },
    {
        "n": 5,
        "name": "isomorphism_tests",
        "label": "Isomorphism & Similarity Tests (all 15 pairs)",
        "module": "isomorphism_tests",
        "entry": "main",
    },
    {
        "n": 6,
        "name": "visualize",
        "label": "Figure Generation (6 publication-ready figures)",
        "module": "visualize",
        "entry": "main",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _sep(char="-", width=70):
    print(char * width)


def _load_and_run(step: dict) -> tuple[bool, float]:
    """
    Dynamically import the step's module and call its entry function.
    Returns (success, elapsed_seconds).
    """
    module_path = os.path.join(_HERE, f"{step['module']}.py")
    spec = importlib.util.spec_from_file_location(step["module"], module_path)
    mod = importlib.util.module_from_spec(spec)
    t0 = time.time()
    try:
        spec.loader.exec_module(mod)
        entry_fn = getattr(mod, step["entry"])
        entry_fn()
        return True, time.time() - t0
    except Exception as exc:
        print(f"\n  [ERROR] Step {step['n']} ({step['name']}) failed:")
        import traceback
        traceback.print_exc()
        return False, time.time() - t0


def run_pipeline(from_step: int = 1, only_step: int = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    _sep("=")
    print("  WP 1.1 -- Emanation Topology Analysis")
    print("  Full Pipeline Orchestrator")
    _sep("=")

    steps_to_run = STEPS
    if only_step is not None:
        steps_to_run = [s for s in STEPS if s["n"] == only_step]
        if not steps_to_run:
            print(f"  ERROR: No step with number {only_step}.")
            sys.exit(1)
        print(f"  Mode: single step {only_step} only")
    else:
        steps_to_run = [s for s in STEPS if s["n"] >= from_step]
        if from_step > 1:
            print(f"  Mode: resuming from step {from_step}")
        else:
            print(f"  Mode: full pipeline ({len(steps_to_run)} steps)")
    _sep()
    print()

    results = []
    pipeline_start = time.time()

    for step in steps_to_run:
        print()
        _sep("-")
        print(f"  STEP {step['n']}/6  --  {step['label']}")
        _sep("-")

        success, elapsed = _load_and_run(step)
        status = "DONE" if success else "FAILED"
        results.append({
            "step": step["n"],
            "name": step["name"],
            "status": status,
            "elapsed": elapsed,
        })

        print()
        print(f"  --> Step {step['n']} {status}  ({elapsed:.2f}s)")

        if not success:
            print()
            print("  Pipeline halted at failed step. Fix the error and re-run with:")
            print(f"    python run_pipeline.py --from {step['n']}")
            break

    # Summary
    total = time.time() - pipeline_start
    print()
    _sep("=")
    print("  PIPELINE SUMMARY")
    _sep("=")
    print(f"  {'Step':<4}  {'Module':<28}  {'Status':<8}  Time")
    _sep()
    for r in results:
        tick = "[OK]" if r["status"] == "DONE" else "[!!]"
        print(f"  {r['step']:<4}  {r['name']:<28}  {tick} {r['status']:<4}  {r['elapsed']:.2f}s")

    n_ok = sum(1 for r in results if r["status"] == "DONE")
    n_total = len(results)
    _sep()
    print(f"  {n_ok}/{n_total} steps completed  |  Total time: {total:.2f}s")

    if n_ok == len(STEPS):
        print()
        print("  ALL 6 PIPELINE STEPS COMPLETE.")
        print()
        print("  Outputs:")
        outputs = [
            ("Schemas (6 JSON)",       "data/schemas/"),
            ("Invariants (7 JSON)",    "outputs/invariants/"),
            ("Controls (8 JSON)",      "outputs/invariants/controls/"),
            ("Statistics (JSON)",      "outputs/invariants/statistical_results.json"),
            ("Similarity matrices",    "outputs/similarity_matrix/"),
            ("Figures (6 PNG, 300dpi)","outputs/figures/"),
        ]
        for label, path in outputs:
            full = os.path.join(_HERE, "..", path)
            exists = os.path.exists(full)
            mark = "+" if exists else "?"
            print(f"  [{mark}] {label:<30}  {path}")
        print()
        print("  Next: encode stretch-goal traditions (Samkhya, Taoist)")
        print("        then run: python paper_draft.py  (or use /paper-draft skill)")

    _sep("=")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WP 1.1 Pipeline Orchestrator"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from this step number (default: 1)"
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None,
        help="Run only this step number"
    )
    args = parser.parse_args()
    run_pipeline(from_step=args.from_step, only_step=args.only_step)


if __name__ == "__main__":
    main()

# WP 1.5 -- Tier 2 Expansion: Vedic, Apocalyptic, and Islamic Traditions in the Emanation Topology Corpus (22 Traditions)

**Awen Grid Empirical Programme -- Work Package 1.5**
**Authors:** Erydir Ceisiwr + Lumos Aureon
**Date:** 2026-04-01

## Summary

WP 1.5 expands the emanation-topology corpus from 17 to 22 traditions, adding five schemas from Vedic, Christian apocalyptic, Islamic, and Jewish apocalyptic sources. Three new methodological analyses are introduced: edge-weighted GED sensitivity across four cost matrices, direction testing for branching trees, and sub-cluster stability tracking. The headline result is the first intra-tradition bifurcation in the corpus: the Rig Veda contains both a linear chain (Nasadiya Sukta) and a branching tree (Purusha Sukta), demonstrating that topology family assignment is determined by cosmological structure, not by tradition, culture, or geography.

## Key Findings

- **22 schemas**: 15 linear chains + 7 branching trees
- **Two-family attractor confirmed** at 22 traditions (separation ratio 2.19x; asymmetric — see below)
- **Intra-tradition bifurcation**: Vedic Nasadiya (linear chain, d4) and Vedic Purusha (branching tree, Br=4) from the same Rig Veda — first case in the corpus
- **Islamic Mi'raj**: deepest linear chain in the corpus (depth 9, z=3.08, p=0.005)
- **Direction asymmetry**: 7/7 branching trees are direction-sensitive (GED > 0 under edge reversal); linear chains are structurally invariant under reversal
- **Cost-matrix robustness**: all structural isomorphisms resolved under all 4 cost matrices (Primary, Uniform, Steep, Compressed)
- **New isomorphism**: Revelation–Bundahishn Zoroastrian (resolved by weighted GED = 2.0–3.0 across matrices)
- **Sub-cluster silhouette**: 0.60 (strong; compact N≤6 = 8 members; deep N≥7 = 7 members)
- **Family asymmetry**: intra-branch mean GED (8.00) > inter-family GED (7.68); separation ratio against branching-tree intra-GED = 0.96x. The two families are not symmetric clusters. The linear-chain family is a tight attractor; the branching-tree family is defined by structural criterion (branching factor > 1), not pairwise proximity.

## Corpus (22 Traditions)

### Linear Chain Family (15)
| Tradition | Nodes | Depth | Source |
|---|---|---|---|
| Plotinian Neoplatonic | 5 | 4 | Enneads |
| Taoist DDJ | 5 | 4 | Dao De Jing |
| Derveni Orphic | 5 | 4 | Derveni Papyrus |
| **Vedic Nasadiya (RV 10.129)** | **5** | **4** | **Rig Veda** |
| Bundahishn Zoroastrian | 6 | 5 | Greater Bundahishn |
| Chaldean | 6 | 5 | Chaldean Oracles |
| **Book of Revelation** | **6** | **5** | **Rev 4--5** |
| **1 Enoch Watchers** | **6** | **5** | **1 En 1--36** |
| Ishraq Illuminationist | 7 | 6 | Hikmat al-Ishraq |
| Lurianic Kabbalistic | 7 | 6 | Etz Chaim |
| Samkhya | 7 | 6 | Samkhya Karika |
| Proclean Neoplatonic | 8 | 7 | Elements of Theology |
| Sethian Gnostic | 8 | 7 | Apocryphon of John |
| Gospel of Mary | 9 | 8 | BG 8502,1 |
| **Islamic Mi'raj** | **10** | **9** | **Sahih al-Bukhari** |

### Branching Tree Family (7)
| Tradition | Nodes | Depth | Max Branch | Source |
|---|---|---|---|---|
| Trimorphic Protennoia | 7 | 4 | 3 | NHC XIII,1 |
| Hermetic | 8 | 3 | 3 | Corpus Hermeticum |
| Genesis Creationist | 8 | 2 | 6 | Genesis 1--2 |
| **Vedic Purusha (RV 10.90)** | **7** | **3** | **4** | **Rig Veda** |
| Valentinian Gnostic | 9 | 7 | 2 | Irenaeus/Ptolemy |
| Manichaean | 11 | 5 | 3 | Kephalaia |
| Popol Vuh (K'iche' Maya) | 11 | 6 | 4 | Popol Vuh |

*New WP 1.5 schemas in **bold**.*

## Pipeline

```bash
# Run full 6-step pipeline
python scripts/run_pipeline.py

# Run sensitivity analysis (cost matrices + direction testing)
python scripts/sensitivity_analysis.py
```

### Pipeline Steps
1. `encode_schemas.py` -- Load + validate 22 schema JSONs against the seven-rule DAG contract
2. `compute_invariants.py` -- 30+ topological metrics per schema
3. `generate_controls.py` -- 22,000 random DAG trees (1,000 pooled + 1,000 per tradition)
4. `statistical_comparison.py` -- 6 formal tests (z-scores, permutation, binomial, Mann-Whitney)
5. `isomorphism_tests.py` -- VF2, GED (structural + role + weighted), WL similarity, subgraph
6. `visualize.py` -- 6 publication-ready figures (300 DPI)

## Directory Structure

```
WP_1.5_Tier2_Traditions/
  data/schemas/           22 tradition JSON files
  scripts/                9 Python pipeline scripts (incl. sensitivity_analysis.py)
  outputs/
    figures/              6 publication-ready PNGs
    invariants/           Per-tradition + aggregated metrics + statistical_results.json
      controls/           22,000 null-model DAG trees
    similarity_matrix/    Pairwise comparison matrices (structural, role, weighted GED; WL; subgraph)
    sensitivity_results.json  Cost-matrix + direction sensitivity outputs
  notes/                  Encoding rationale documents
  README.md               This file
  WP1.5_handoff.md        Task specification and encoding guidance
```

## Dependencies

- Python 3.10+
- networkx, numpy, scipy, matplotlib

## Methodological Innovations (WP 1.5)

1. **Edge-weighted GED sensitivity analysis**: Four cost matrices tested (Primary 0/0.5/1/1.5, Uniform, Steep, Compressed). All structural isomorphisms resolved under all four matrices.

2. **Direction testing for branching trees**: For each of the 7 branching trees, all edges reversed and structural GED computed. All 7/7 produce non-zero GED (direction carries structural information). Linear chains are structurally invariant under reversal — direction matters for trees, not chains.

3. **Sub-cluster stability tracking**: Compact (N≤6) and deep (N≥7) sub-clusters re-evaluated at 22 schemas. Silhouette 0.63 → 0.60 (strong clustering maintained).

4. **Intra-tradition bifurcation test**: Both Vedic hymns encoded and classified independently. First demonstration that a single textual corpus contains both topology families.

## Previous Work Packages

| WP | Schemas | Repo | DOI |
|---|---|---|---|
| 1.1 | 9 | [emanation-topology](https://github.com/OwainGlyndwr1400/emanation-topology) | pending |
| 1.2 | 11 | [corpus-expansion-emanation](https://github.com/OwainGlyndwr1400/corpus-expansion-emanation) | [10.5281/zenodo.19305988](https://doi.org/10.5281/zenodo.19305988) |
| 1.3 | 14 | [structural-attractors-emanation-cosmologies-wp1-3](https://github.com/OwainGlyndwr1400/structural-attractors-emanation-cosmologies-wp1-3) | [10.5281/zenodo.19324327](https://doi.org/10.5281/zenodo.19324327) |
| 1.4 | 17 | [geographic-generality-emanation-cosmologies-wp1-4](https://github.com/OwainGlyndwr1400/geographic-generality-emanation-cosmologies-wp1-4) | [10.5281/zenodo.19340999](https://doi.org/10.5281/zenodo.19340999) |
| **1.5** | **22** | **this repo** | [**10.5281/zenodo.19362550**](https://doi.org/10.5281/zenodo.19362550) |

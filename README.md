# Tier 2 Expansion — Vedic, Apocalyptic and Islamic Traditions

**One text, two families. The Rig Veda contains both a linear chain and a
branching tree — which is the closest thing this series has to a controlled
experiment.**

Every result so far could be explained away by culture: perhaps Greek cosmologies
are chains because Greeks think in chains. Work Package 1.5 removes that
explanation. The **Nasadiya Sukta** and the **Purusha Sukta** sit in the *same
scripture*, in the *same language*, from the *same culture*, in the *same
period* — and they land in opposite topology families. Nasadiya is a linear chain
of depth 4. Purusha is a branching tree with branching factor 4.

Culture, language and geography are held constant. The topology still splits.
Whatever the graph is tracking, it is a property of the cosmological structure
itself.

Corpus: 22 traditions, adding five Vedic, Christian apocalyptic, Islamic and
Jewish apocalyptic schemas.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19362550.svg)](https://doi.org/10.5281/zenodo.19362550)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Licence](https://img.shields.io/badge/Licence-MIT-green)](#licence)

*Work Package 1.5 of the Awen Grid Empirical Programme.*

---

## Run it

```bash
git clone https://github.com/OwainGlyndwr1400/tier2-expansion-emanation-cosmologies-wp1-5.git
cd tier2-expansion-emanation-cosmologies-wp1-5
pip install -r requirements.txt

python scripts/run_pipeline.py           # full 6-step pipeline
python scripts/sensitivity_analysis.py   # 4 cost matrices + direction testing
```

Python 3.11+.

---

## Results

- **22 schemas:** 15 linear chains, 7 branching trees.
- **Intra-tradition bifurcation** — Vedic Nasadiya (linear chain, depth 4) and
  Vedic Purusha (branching tree, Br = 4), both from the Rig Veda. First case in
  the corpus.
- **Islamic Mi'raj:** deepest linear chain in the corpus — depth 9,
  z = 3.08, **p = 0.005**.
- **Direction asymmetry:** all **7/7** branching trees are direction-sensitive
  (GED > 0 under edge reversal), while linear chains are structurally invariant
  under reversal. The two families do not merely differ in shape; they differ in
  how they respond to reversal.
- **Cost-matrix robustness:** every structural isomorphism resolves under all
  **four** cost matrices tested (Primary, Uniform, Steep, Compressed). The
  isomorphisms are not an artefact of one weighting choice.
- **New isomorphism:** Revelation ↔ Bundahishn Zoroastrian, resolved by weighted
  GED at 2.0–3.0 across matrices.
- **Sub-cluster silhouette:** 0.60 — compact N ≤ 6 (8 members), deep N ≥ 7 (7 members).

### The result that complicates the story

**The two families are not symmetric clusters.** Intra-branching-tree mean GED
(8.00) is *greater* than inter-family GED (7.68), giving a separation ratio of
**0.96x** measured against branching-tree intra-distance.

Stated plainly: the linear-chain family is a genuine tight attractor. The
branching-tree family is not — it is a residual category, defined by a structural
criterion (branching factor > 1) rather than by its members being close to one
another. Two branching trees can be further apart than a branching tree and a
chain.

That finding weakens the neat "two attractors" framing carried since WP 1.1, and
it is reported here rather than buried, because it is what the data says.

---

## Method additions

Three new analyses in this work package:

1. **Edge-weighted GED sensitivity** across four cost matrices — tests whether
   isomorphism results depend on the weighting scheme. They do not.
2. **Direction testing for branching trees** — GED recomputed under full edge
   reversal.
3. **Sub-cluster stability tracking** — carries the compact/deep split forward
   across corpus growth so it can be checked for drift.

## The series

| WP | Traditions | Repository | DOI | Headline |
|---|---|---|---|---|
| 1.1 | 9 | [emanation-topology](https://github.com/OwainGlyndwr1400/emanation-topology) | pending | Method established; 2 exact isomorphisms |
| 1.2 | 11 | [corpus-expansion-emanation](https://github.com/OwainGlyndwr1400/corpus-expansion-emanation) | [zenodo.19305988](https://doi.org/10.5281/zenodo.19305988) | Proclus + Suhrawardi; isomorphisms rise to 5 |
| 1.3 | 14 | [structural-attractors (wp1-3)](https://github.com/OwainGlyndwr1400/structural-attractors-emanation-cosmologies-wp1-3) | [zenodo.19324327](https://doi.org/10.5281/zenodo.19324327) | Zoroastrian, Manichaean, Orphic; separation peaks at 2.87x |
| 1.4 | 17 | [geographic-generality (wp1-4)](https://github.com/OwainGlyndwr1400/geographic-generality-emanation-cosmologies-wp1-4) | [zenodo.19340999](https://doi.org/10.5281/zenodo.19340999) | Popol Vuh - zero Old World contact, same families |
| **1.5** | **22** | **this repo** | **[zenodo.19362550](https://doi.org/10.5281/zenodo.19362550)** | **The Rig Veda splits across *both* families** |
| 1.6 | 25 | [geographic-role-expansion (wp1-6)](https://github.com/OwainGlyndwr1400/geographic-role-expansion-emanation-cosmologies-wp1-6) | [zenodo.19368287](https://doi.org/10.5281/zenodo.19368287) | Five-way zero-contact convergence |

## Citation

> Ceisiwr, Erydir, and Lumos Aureon. *WP 1.5 — Tier 2 Expansion: Vedic,
> Apocalyptic, and Islamic Traditions in the Emanation Topology Corpus
> (22 Traditions).* Awen Grid Empirical Programme, 2026.
> [10.5281/zenodo.19362550](https://doi.org/10.5281/zenodo.19362550)

## Licence

MIT — code and data freely reusable with attribution.

## Author

Erydir Ceisiwr — Independent Researcher, Awen Grid Programme, Swansea, Wales.
ORCID [0009-0004-4577-5253](https://orcid.org/0009-0004-4577-5253)

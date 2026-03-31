# WP 1.5 Encoding Rationale Notes

## Schema Summary

| Schema | Tradition | N | D | Br | Family | Edge Types | Flags |
|---|---|---|---|---|---|---|---|
| vedic_nasadiya | Rig Veda 10.129 | 5 | 4 | 1 | linear chain | emanation(2), creation(2) | — |
| vedic_purusha | Rig Veda 10.90 | 7 | 3 | 4 | branching tree | emanation(1), fragmentation(1), creation(4) | proc+frag |
| book_of_revelation | Apocalypse of John | 6 | 5 | 1 | linear chain | emanation(3), creation(2) | — |
| quranic_miraj | Mi'raj (Bukhari) | 10 | 9 | 1 | linear chain | emanation(8), creation(1) | — |
| enochian_watchers | 1 Enoch 1-36 | 6 | 5 | 1 | linear chain | emanation(2), creation(2), fragmentation(1) | fall+frag |

## Key Encoding Decisions

### 1. vedic_nasadiya — Tad Ekam as source
- "That One" (tad ekam) is ambiguous — the hymn negates both being and non-being
- Encoded as source because functionally it is the origin of the cosmogonic sequence
- The hymn's agnosticism (verses 6-7) is noted but does not affect topology
- Edge pattern: emanation → emanation → creation → creation (same as Taoist DDJ)

### 2. vedic_purusha — Sacrifice as process node with 4 branches
- Yajna (sacrifice) encoded as process, not an entity — the dismemberment IS the cosmogonic act
- Four branches at sacrifice level: varnas, cosmic elements, ritual elements, animals
- Each branch encoded as a single node (not expanded to individual items) for cross-tradition comparability
- Fragmentation edge (Viraj → Yajna) = positive/generative dismemberment, first in corpus
- Animals kept as separate branch (not merged with cosmic elements) because textually distinct

### 3. book_of_revelation — Cosmic hierarchy, not eschatological drama
- Seven Seals/Trumpets/Bowls are EVENTS, not levels — excluded from encoding
- Throne-room hierarchy (Rev 4-5) provides the cosmological structure
- Seven Spirits encoded as single intermediary node (functional unity)
- New Jerusalem as redemptive matter — inverts the typical emanation degradation pattern
- Structural isomorphism with Bundahishn Zoroastrian (both 6n, d5) — resolved by weighted GED

### 4. quranic_miraj — Bukhari sequence as primary
- Each heaven is a separate node because each has a distinct prophet-guardian
- Sidrat al-Muntaha encoded as intermediary (not new "boundary" role — agreed with Erydir)
- Sahih al-Bukhari prophet ordering used (most widely attested)
- Deepest chain in corpus: depth 9, surpassing Gospel of Mary (depth 8)
- 8/9 edges are emanation — most emanation-heavy schema in corpus

### 5. enochian_watchers — Book of the Watchers only
- 1 Enoch chapters 1-36 (not mixed with Parables or Luminaries)
- Watcher fall encoded as fragmentation edge (gate_of_heaven → watchers_fallen)
- Gate of Heaven encoded as intermediary (not new "boundary" role)
- 2 Enoch seven-heavens noted as alternative encoding for future WP

## Boundary Role Decision
- Discussed with Erydir: "boundary" role NOT added for WP 1.5
- Sidrat al-Muntaha and Gate of Heaven encoded as "intermediary"
- Reason: binary role substitution cost (0/1) makes the distinction analytically insignificant
- Revisit if/when graded role-substitution costs are implemented (WP 1.6 candidate)

## Intra-Tradition Bifurcation
- vedic_nasadiya (linear chain, 5n, d4) and vedic_purusha (branching tree, 7n, d3, Br=4)
- Same Rig Veda Book 10, same approximate date (c. 1200-900 BCE)
- Different cosmological strategies: process-agnostic vs sacrificial-hierarchical
- Confirms that topology family is determined by cosmological STRUCTURE, not by tradition/culture
- First demonstration in the corpus

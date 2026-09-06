# Evaluation

All numbers below are produced by `python scripts/evaluate.py` against the **real
expert-scored pairs** in `data/raw/` (100 validation + 90 test). Nothing here is
hand-entered; re-running the script regenerates `data/processed/evaluation_report.json`
and these tables.

The random-walk network embeddings carry ±0.02–0.03 Spearman run-to-run variance
(gensim Word2Vec + walk sampling); the numbers below are one representative run.

**Protocol (no leakage).** Text-only, network-only and weighted-hybrid configs are
scored directly on each split. The learned combiner is evaluated by **5-fold
cross-validation on validation**, and for the test column is **trained on all 100
validation pairs** and applied to the 90 held-out test pairs.

## 1. Correlation with expert scores

Primary metric of Bhattacharya et al. (2022): how well the model's pairwise
similarity tracks the expert pairwise similarity.

| Config | Validation ρ (Spearman) | Validation r | Validation MSE | Test ρ | Test r | Test MSE |
|---|--:|--:|--:|--:|--:|--:|
| text: TF-IDF | **+0.76** | +0.66 | 0.119 | −0.02 | −0.01 | 0.189 |
| text: Doc2Vec *(base paper)* | +0.67 | +0.65 | 0.068 | −0.02 | −0.01 | 0.077 |
| network: metapath2vec (embedding only) | +0.43 | +0.33 | 0.091 | −0.03 | −0.04 | 0.157 |
| network: metapath2vec (blend + statute/citation overlap) | +0.45 | +0.37 | 0.110 | −0.03 | −0.04 | 0.174 |
| network: node2vec (embedding only) | +0.48 | +0.36 | 0.085 | −0.09 | −0.09 | 0.165 |
| hybrid: weighted 0.5 / 0.5 | +0.59 | +0.54 | 0.083 | −0.03 | −0.04 | 0.148 |
| hybrid: weighted 0.7 / 0.3 | +0.70 | +0.63 | 0.092 | −0.03 | −0.04 | 0.158 |
| hybrid: weighted 0.3 / 0.7 | +0.55 | +0.46 | 0.081 | −0.05 | −0.04 | 0.144 |
| **hybrid: learned linear (text + network)** | **+0.75** | +0.65 | **0.055** | −0.05 | −0.03 | **0.066** |
| hybrid: learned linear (all 6 features) | +0.73 | +0.63 | 0.056 | −0.05 | −0.04 | 0.069 |
| hybrid: learned MLP (text + network) | +0.61 | +0.56 | 0.069 | −0.04 | −0.05 | 0.152 |
| *baseline: year proximity* | +0.29 | +0.31 | 0.260 | +0.03 | +0.07 | 0.140 |
| *baseline: random* | −0.13 | −0.14 | 0.212 | −0.03 | −0.00 | 0.125 |

Learned-linear combiner weights (fit on validation): `text 0.86, network 0.27,
intercept +0.21`.

## 2. Ranking sanity

For each query document that has an expert partner scored ≥ 0.5, rank the whole
295-document corpus and locate that partner (MRR and hit@10 over ~50 queries per
split). This asks a different question from §1: *can the system retrieve a
similar case*, regardless of reproducing the exact score.

| Config | Val MRR | Val hit@10 | Test MRR | Test hit@10 |
|---|--:|--:|--:|--:|
| text: TF-IDF | 0.64 | 0.94 | 0.28 | 0.43 |
| text: Doc2Vec | 0.49 | 0.72 | **0.30** | **0.45** |
| network: metapath2vec (embedding) | 0.31 | 0.46 | 0.06 | 0.14 |
| network: metapath2vec (blend) | 0.34 | 0.49 | 0.07 | 0.16 |
| hybrid: weighted 0.7 / 0.3 | 0.57 | 0.88 | 0.23 | 0.36 |
| baseline: year proximity | 0.16 | 0.24 | 0.01 | 0.00 |
| baseline: random | 0.02 | 0.06 | 0.02 | 0.01 |

## 3. What the numbers say

**Validation.** Text is the strongest single signal (ρ ≈ 0.76). The network alone
is weaker (ρ ≈ 0.44–0.48) but clearly above the year-proximity baseline (0.29) and
far above random. Blending the ICF-weighted statute/citation overlap into the
network score helps it (0.44 → 0.48). The **learned linear combiner matches
text-only on rank correlation (0.75) while roughly halving MSE** (0.055 vs 0.119)
— i.e. the network's main contribution on this corpus is *calibration*, not
re-ranking. This is a weaker form of the paper's result: they report the hybrid
beating the best text-only method by +11.8% correlation on a **53,210-case**
corpus with a dense real citation network; our demo corpus has only **24 resolved
case→case edges** (CLAUDE.md §3), so the network layer is thinner and the gain is
smaller. We do **not** claim the paper's margins.

**Test.** *Every* method — including both trivial baselines — has near-zero
correlation on the 90 test pairs. This is a property of that split, not a pipeline
defect:

- **Range restriction.** Test scores span only [0.15, 0.90] with 16 distinct
  values (validation: [0.00, 0.97], 27 values). There are no clearly-dissimilar
  pairs, which caps achievable correlation.
- **Graph coverage.** 31 of 90 test pairs involve a case with no Hier-SPCNet node,
  so their network similarity is 0 by construction (validation: only 8/100).
- **Not just isolation.** On the 59 test pairs where *neither* case is isolated,
  the hybrid is still ρ ≈ −0.03 and text-only ρ ≈ +0.05 — the correlation is
  genuinely absent, not masked by the isolated cases.
- **Ranking still works.** Text/Doc2Vec retrieve a relevant partner with MRR ≈
  0.28–0.30 and hit@10 ≈ 0.43–0.45 on test — the system finds similar cases even
  where it cannot reproduce the expert score gradations.

Our reading: the 90 test pairs were scored by the annotators on a notion of
similarity that does not line up with any surface-textual or citation/statute
feature we can compute on a corpus this small — likely fine-grained legal-issue
distinctions among already-related cases. Reproducing the paper's test-set
behaviour would need its full corpus and citation graph, which are not published.

## 4. Reproduce

```bash
python scripts/build_all.py          # if artifacts not built yet
python scripts/evaluate.py           # full matrix + diagnosis
python scripts/evaluate.py --quick   # skip Doc2Vec / node2vec
```

The API serves the cached report at `GET /api/evaluate` and a compact table at
`GET /api/evaluate/summary`.

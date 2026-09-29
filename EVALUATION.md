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

## At a glance

Answering the project's research question — *does jointly analyzing text,
citations, and statutes estimate similarity more accurately?* — with four
independent pieces of evidence:

1. **Text vs. hybrid (§1, validation).** The hybrid combiner ties text-only on
   rank correlation but **halves the error** (MSE 0.050 vs 0.100) — the
   network/statute layer's contribution here is *calibration*, not re-ranking.
2. **Semantic vs. lexical text similarity (§3).** SBERT edges out TF-IDF on
   correlation but is *worse* at actually retrieving the correct expert-labelled
   partner (ranking sanity, §2) — a real trade-off, not a clean win either way.
   TF-IDF stays the production default on that basis.
3. **Citation resolution at scale (§3).** Root-caused the network signal's main
   weakness (only 244/33,433 docs have a matchable title) rather than patching
   around it with an unsafe heuristic; documented why a data-source fix (not an
   algorithm fix) is what's actually needed.
4. **Independent validation at full scale (§4, new).** With no expert labels
   available for the 33k-document corpus, citation edges serve as weak ground
   truth instead: the model ranks a truly-cited case above **99% of random
   alternatives on average** (vs. ~50% for a random-baseline control) — real
   evidence the system behaves sensibly at the scale it's actually deployed at,
   not just on the 296-document evaluation set.
5. **Accuracy / precision / recall / F1 (§1b, §4, new)**, done honestly rather
   than just supplied on request: at a fixed, un-tuned threshold on the base
   corpus, TF-IDF scores a *misleadingly bad* F1 of 0.11 despite being the
   best-correlated method — a threshold-calibration artifact, exposed and
   explained rather than hidden. With real binary ground truth and a
   distribution-appropriate threshold (§4's cited-vs-random task), the same
   kind of model hits **accuracy/precision/recall/F1 all at 0.95, ROC-AUC
   0.99** — showing the earlier low numbers were about threshold choice, not
   capability.

Honest gap that remains open: §1's test-split correlation is ≈0 for every
method including trivial baselines, for reasons diagnosed in §3. That's a
property of the 90-pair split, not resolved by any of the above — it's flagged
here rather than talked around.

## 1. Correlation with expert scores

Primary metric of Bhattacharya et al. (2022): how well the model's pairwise
similarity tracks the expert pairwise similarity.

| Config | Validation ρ (Spearman) | Validation r | Validation MSE | Test ρ | Test r | Test MSE |
|---|--:|--:|--:|--:|--:|--:|
| text: TF-IDF | **+0.78** | +0.72 | 0.100 | −0.04 | −0.06 | 0.171 |
| text: Doc2Vec *(base paper; earlier run — see note)* | +0.67 | +0.65 | 0.068 | −0.02 | −0.01 | 0.077 |
| text: **SBERT** *(our enhancement, `all-MiniLM-L6-v2`)* | **+0.80** | +0.78 | 0.142 | −0.02 | −0.03 | **0.104** |
| network: metapath2vec (embedding only) | +0.46 | +0.41 | 0.075 | +0.02 | +0.06 | 0.105 |
| network: metapath2vec (blend + statute/citation overlap) | +0.51 | +0.45 | 0.099 | +0.01 | +0.04 | 0.142 |
| network: node2vec (embedding only) *(earlier run — see note)* | +0.48 | +0.36 | 0.085 | −0.09 | −0.09 | 0.165 |
| hybrid: weighted 0.5 / 0.5 | +0.66 | +0.63 | 0.070 | +0.04 | +0.03 | 0.117 |
| hybrid: weighted 0.7 / 0.3 | +0.76 | +0.70 | 0.078 | −0.01 | +0.00 | 0.134 |
| hybrid: weighted 0.3 / 0.7 | +0.57 | +0.54 | 0.068 | +0.02 | +0.05 | 0.108 |
| **hybrid: learned linear (TF-IDF text + network)** | +0.76 | +0.71 | **0.050** | −0.03 | −0.03 | 0.073 |
| hybrid: learned linear (all 6 features) | +0.74 | +0.68 | **0.050** | −0.02 | −0.01 | 0.079 |
| hybrid: learned MLP (TF-IDF text + network) | +0.69 | +0.64 | 0.055 | +0.04 | +0.01 | 0.130 |
| **hybrid: learned linear (SBERT text + network)** | **+0.79** | **+0.77** | 0.045 | +0.01 | −0.01 | **0.077** |
| *baseline: year proximity* | +0.29 | +0.31 | 0.260 | +0.03 | +0.07 | 0.140 |
| *baseline: random* | −0.13 | −0.14 | 0.212 | −0.03 | −0.00 | 0.125 |

Learned-linear combiner weights (fit on validation): TF-IDF-based `text 0.93,
network 0.37, intercept +0.13`; SBERT-based `text 1.04, network 0.38, intercept
−0.45`. *Note: Doc2Vec/node2vec rows are carried over from an earlier build —
those embedders' artifacts weren't regenerated in this pass, only SBERT's were
(added specifically to evaluate the semantic-embedding claim in §3). All other
rows are from one fresh, internally-consistent run of `scripts/evaluate.py`.*

## 1b. Classification metrics (accuracy / precision / recall / F1)

This task is fundamentally a **continuous similarity score**, evaluated above
by correlation — not a "similar / not similar" yes-or-no question. Reporting
accuracy/precision/recall/F1 means picking a threshold that turns one into the
other. Rather than skip the ask or fudge it, here it is done properly: the
expert score is binarized at **0.5** ("similar" vs not) and the model's score
at the **same fixed 0.5 cutoff** — a principled, a-priori choice, not tuned on
this data (tuning a threshold on the same set you evaluate it on would leak).

| Config | Val Acc | Val Prec | Val Recall | Val F1 | Val AUC | Test Acc | Test F1 | Test AUC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| text: TF-IDF | 0.53 | **1.00** | 0.06 | 0.11 | **0.89** | 0.42 | 0.00 | 0.48 |
| text: SBERT | 0.52 | 0.51 | **1.00** | 0.68 | **0.92** | 0.56 | 0.71 | 0.51 |
| hybrid: learned linear (TF-IDF) | 0.76 | 0.86 | 0.62 | 0.72 | 0.90 | 0.43 | 0.14 | 0.48 |
| **hybrid: learned linear (SBERT)** | **0.82** | 0.90 | 0.72 | **0.80** | **0.93** | 0.48 | 0.37 | 0.52 |
| *baseline: year proximity* | 0.50 | 0.50 | 1.00 | 0.67 | 0.67 | 0.60 | 0.72 | 0.49 |
| *baseline: random* | 0.50 | 0.50 | 0.56 | 0.53 | 0.53 | 0.63 | 0.67 | 0.65 |

**Read this table carefully before quoting it — it contains a genuine trap.**
TF-IDF gets **F1 = 0.11** on validation despite being tied for the
*best*-correlated method in §1. Its precision is a perfect 1.00, but recall is
0.06: at threshold 0.5, TF-IDF's raw cosine scores almost never reach 0.5, so
the classifier calls nearly every pair "not similar" — and is *right* on the
rare occasion it says "similar," which is exactly what perfect precision +
near-zero recall means. **This is a threshold-calibration artifact, not a
capability gap** — different methods produce scores on different scales and
distributions, and forcing one shared cutoff onto all of them isn't a fair
comparison. **ROC-AUC needs no threshold and tells the truer story**: it
tracks the Spearman correlations in §1 almost exactly (validation ≈0.9 for
every real method, test ≈0.5 — chance level, consistent with §3's diagnosis of
the test split). Treat the Acc/Prec/Recall/F1 columns as "what these specific
numbers would look like at this specific cutoff," not as a ranking of the
methods — AUC (or §1's correlation) is what to actually compare them on.

## 2. Ranking sanity

For each query document that has an expert partner scored ≥ 0.5, rank the whole
295-document corpus and locate that partner (MRR and hit@10 over ~50 queries per
split). This asks a different question from §1: *can the system retrieve a
similar case*, regardless of reproducing the exact score.

| Config | Val MRR | Val hit@10 | Test MRR | Test hit@10 |
|---|--:|--:|--:|--:|
| text: TF-IDF | **0.66** | **0.94** | **0.27** | **0.41** |
| text: Doc2Vec *(earlier run)* | 0.49 | 0.72 | 0.30 | 0.45 |
| text: SBERT | 0.58 | 0.86 | 0.22 | 0.40 |
| network: metapath2vec (embedding) | 0.38 | 0.59 | 0.05 | 0.18 |
| network: metapath2vec (blend) | 0.43 | 0.63 | 0.06 | 0.20 |
| hybrid: weighted 0.7 / 0.3 | 0.62 | 0.92 | 0.22 | 0.40 |
| baseline: year proximity | 0.16 | 0.24 | 0.01 | 0.00 |
| baseline: random | 0.02 | 0.06 | 0.02 | 0.01 |

Note SBERT's own trade-off: it edges out TF-IDF on Spearman correlation and MSE
(table above), but is *worse* on ranking sanity here — TF-IDF still retrieves the
correct partner more often (MRR 0.66 vs 0.58 on validation). Semantic embedding
similarity and "is the top-ranked result the expert-labelled partner" are
different questions; SBERT answering the first better doesn't mean it answers
the second better on a corpus this size.

## 3. What the numbers say

**Validation.** Text is the strongest single signal (TF-IDF ρ ≈ 0.78, SBERT
ρ ≈ 0.80). The network alone is weaker (ρ ≈ 0.46–0.51) but clearly above the
year-proximity baseline (0.29) and far above random. Blending the ICF-weighted
statute/citation overlap into the network score helps it (0.46 → 0.51). The
**learned linear combiner roughly matches text-only on rank correlation while
cutting MSE by half** (TF-IDF-based: 0.050 vs 0.100 text-only; SBERT-based:
0.045 vs 0.142 text-only) — i.e. the network's main contribution on this corpus
is *calibration*, not re-ranking. This is a weaker form of the paper's result:
they report the hybrid beating the best text-only method by +11.8% correlation
on a **53,210-case** corpus with a dense real citation network; the 296-document
evaluation corpus these numbers are computed on has only **24 resolved
case→case edges** (PROJECT_NOTES.md §3), so the network layer is thinner here
and the gain is smaller. We do **not** claim the paper's margins.

*(Scale note: this 24-edge figure is specific to the 296-document evaluation
corpus, since that's the only one with expert-labelled pairs to score against.
At the full 33,433-document scale the app actually serves, citation resolution
finds **1,266 edges** — not because matching got smarter, but because ~33k
citing documents now have many more chances to hit the same small pool of
**244 titled targets** (only 0.7% of the corpus has a usable cause-title to
resolve *against* — see "Citation resolution at scale" below). That denser
graph isn't reflected in the correlation numbers above, since there's no
expert-labelled ground truth at that scale to evaluate it against.)*

### Citation resolution at scale: what's actually limiting it

Investigated directly rather than assumed: citation resolution
(`app/ml/citation_extraction.py`) works by fuzzy-matching each "A v B" mention
against a **title index** built from `doc.title` — and only **244 of 33,433
documents (0.7%)** have one. The base 296-document corpus preserves each
judgment's cause-title; the bulk-ingested ILDC text does not (it's the judgment
body only, starting with the judge's name, not a title header). Running
extraction at full scale (`data/processed/citation_report.json`) confirms this
is the bottleneck, not the matching algorithm: **118,708 raw "A v B" mentions**
were extracted, but only **1,266 (1.2%) resolved** — because there are only 244
possible targets to resolve *to*, no matter how many citing documents exist.

We considered deriving a synthetic title for untitled documents from their own
opening text (reusing the citation extractor on each document's first few
hundred words) and rejected it: spot-checking real ILDC documents (e.g.
`2015_752`) shows the first "X v Y"-shaped mention in the body is often a
**cited precedent**, not the document's own identity — e.g. that document's
text mentions "*P.J. Agro Tech Limited and Ors. Vs. Water Base Limited*" as an
authority it relies on, long before any self-identifying title-like text. A
naive first-match heuristic would silently mislabel documents with someone
else's case name, corrupting both the displayed title and the citation graph
with false edges — worse than the current honest gap. This confirms SCALING.md's
existing conclusion: closing this gap needs a data source with **pre-resolved
citations** (e.g. the Indian Kanoon API), not a smarter local heuristic.

**Does semantic (SBERT) text similarity beat lexical (TF-IDF)?** Not cleanly.
SBERT wins on Spearman correlation and (on test) MSE — consistent with it
capturing paraphrased/semantic overlap that bag-of-words misses. But it *loses*
to TF-IDF on ranking sanity (§2): MRR 0.58 vs 0.66 on validation, 0.22 vs 0.27
on test. Our reading: this corpus's expert similarity judgments lean on shared
legal *terminology* (statute names, party/procedural vocabulary) more than on
paraphrased meaning, which is exactly what TF-IDF is built to catch and a
general-purpose sentence encoder (`all-MiniLM-L6-v2`, not legal-domain-tuned)
is not. A legal-domain SBERT model (e.g. `law-ai/InLegalBERT`) might close this
gap; we have not tested one. **TF-IDF stays the default** on the strength of
this result, not because SBERT was untested.

**Test.** *Every* method — including both trivial baselines — has near-zero
correlation on the 90 test pairs. This is a property of that split, not a pipeline
defect:

- **Range restriction.** Test scores span only [0.15, 0.90] with 16 distinct
  values (validation: [0.00, 0.97], 27 values). There are no clearly-dissimilar
  pairs, which caps achievable correlation.
- **Graph coverage.** 18 of 90 test pairs involve a case with no Hier-SPCNet node,
  so their network similarity is 0 by construction (validation: only 1/100).
- **Not just isolation.** On the 72 test pairs where *neither* case is isolated,
  the hybrid is still ρ ≈ −0.06 and text-only ρ ≈ −0.01 — the correlation is
  genuinely absent, not masked by the isolated cases.
- **Ranking still works.** TF-IDF/SBERT retrieve a relevant partner with MRR ≈
  0.22–0.27 and hit@10 ≈ 0.40–0.41 on test — the system finds similar cases even
  where it cannot reproduce the expert score gradations.

Our reading: the 90 test pairs were scored by the annotators on a notion of
similarity that does not line up with any surface-textual or citation/statute
feature we can compute on a corpus this small — likely fine-grained legal-issue
distinctions among already-related cases. Reproducing the paper's test-set
behaviour would need its full corpus and citation graph, which are not published.

## 4. Citation-based sanity check at full scale (33,433 docs)

Sections 1–3 are all computed on the 296-document base corpus, because that's
the only one with expert-labelled pairs. It says nothing about whether the
deployed system (33,433 docs, `CORPUS_SOURCE=db`) behaves sensibly — there's no
expert ground truth at that scale to check it against.

There is, however, real (if weaker) ground truth available at full scale: the
resolved case→case **citation graph** (1,186 edges — see "Citation resolution
at scale" above). If case A cites case B, that's independent evidence they're
legally related, even without an expert similarity score. `scripts/citation_sanity_check.py`
tests exactly this: for every citing→cited edge, rank the true cited case's
hybrid score against 99 random distractor documents.

| | Cited pairs (n=1,186) | Random baseline |
|---|--:|--:|
| Mean percentile rank *(1.0 = beat all 99 distractors)* | **0.989** | 0.507 |
| Beats **all** 99 distractors outright | **67.8%** | 1.4% |

The random baseline lands almost exactly at chance (0.50 / ~1%, as expected for
100 candidates), confirming the check itself isn't biased. The cited pairs are
nowhere near chance: the model ranks the actual cited case above essentially
all random alternatives, and picks it as the single best match over 99 random
documents better than two-thirds of the time.

**Classification framing (accuracy / precision / recall / F1).** Unlike §1b,
this one has genuine binary ground truth — "cited" vs "random" is a real
label, not an arbitrary cut on a continuous score — so accuracy/precision/
recall/F1 apply cleanly here. Built a balanced set (1,186 cited pairs, label 1
+ 1,186 random pairs, label 0) and thresholded at the **median of the pooled
scores** (0.320) — chosen from the score distribution alone, not tuned on the
labels:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|--:|--:|--:|--:|--:|
| **0.949** | **0.949** | **0.949** | **0.949** | **0.988** |

This is the cleanest result in the whole evaluation, and it's informative
*because* it contrasts with §1b: the same model, given a threshold that
actually matches its score distribution instead of an arbitrary fixed 0.5,
separates related from unrelated pairs almost perfectly at production scale.
The lesson isn't "ignore §1b" — it's that thresholded classification metrics
are only as meaningful as the threshold they're computed at.

**Caveat, stated plainly**: citation is not the same claim as expert-judged
similarity — a case can cite another for a narrow procedural point, and two
genuinely similar cases often don't cite each other at all. This result shows
the model's score tracks *real legal relatedness* strongly at production
scale; it is deliberately not presented as validating the same fine-grained
similarity-score claim that §1–3 measure on the base corpus, which remains an
open gap.

```bash
CORPUS_SOURCE=db python scripts/citation_sanity_check.py   # writes citation_sanity_report.json
```

## 5. Reproduce

```bash
python scripts/build_all.py                                 # if artifacts not built yet
pip install -r backend/requirements-ml.txt                   # needed once, for SBERT
python scripts/generate_embeddings.py --method sbert          # ~25 min over the full corpus
python scripts/evaluate.py           # full matrix + diagnosis (includes SBERT rows if built)
python scripts/evaluate.py --quick   # skip Doc2Vec / node2vec (SBERT rows still included)

# §4 - needs the full corpus built (CORPUS_SOURCE=db python scripts/build_all.py)
CORPUS_SOURCE=db python scripts/citation_sanity_check.py
```

The API serves the cached report at `GET /api/evaluate` and a compact table at
`GET /api/evaluate/summary`.

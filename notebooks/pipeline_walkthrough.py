# ---
# jupyter:
#   jupytext:
#     text_representation: {format_name: percent}
# ---
# %% [markdown]
# # Pipeline walkthrough
#
# Uses the `app.ml.*` modules directly (no FastAPI). Run from the repo root with
# the project venv. Open in VS Code / Jupyter (the `# %%` markers are cells), or
# just `python notebooks/pipeline_walkthrough.py`.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "backend"))

from app.utils.dataset import load_corpus

corpus = load_corpus()
print(f"{len(corpus)} unique documents, "
      f"{len(corpus.pairs_by_split['validation'])}+{len(corpus.pairs_by_split['test'])} expert pairs")

# %% [markdown]
# ## 1. Preprocessing

# %%
from app.ml.preprocessing import preprocess_text

pd = preprocess_text(corpus.text("2005_S_628"))
print("paragraphs:", pd.stats["n_paragraphs"], " sentences:", pd.stats["n_sentences"],
      " tokens:", pd.stats["n_tokens"])
print(pd.sentences[1][:160])

# %% [markdown]
# ## 2. Citation + statute extraction

# %%
from app.ml.citation_extraction import extract_case_citations
from app.ml.statute_extraction import resolve_document_statutes

cits = extract_case_citations(corpus.text("2005_S_628"))
print("case citations:", [c.raw for c in cits[:5]])

ds = resolve_document_statutes(corpus.text("2005_S_628"))
print("primary Act:", ds.primary_act_id, " Acts:", ds.act_ids[:5])

# %% [markdown]
# ## 3. Text + network + hybrid similarity

# %%
from app.ml.text_similarity import TextSimilarityIndex, TfidfEmbedder
from app.ml.network_embedding import NetworkEmbedder, NetworkEmbeddingIndex
from app.ml.network_model import build_hier_spcnet
from app.ml.network_similarity import NetworkSimilarityIndex
from app.ml.hybrid_similarity import HybridSimilarityModel, LearnedCombiner

ti = TextSimilarityIndex.build(corpus, TfidfEmbedder())
graph = build_hier_spcnet(corpus, min_score=90)
ei = NetworkEmbeddingIndex.build(graph, corpus, NetworkEmbedder(method="metapath2vec"))
nsi = NetworkSimilarityIndex.build(corpus, ei, method="blend")

model = HybridSimilarityModel(ti, nsi, LearnedCombiner(model="linear"))
print(model.fit_learned(corpus, split="validation"))

# %% [markdown]
# ## 4. Recommend + explain

# %%
from app.ml.recommendation import RecommendationEngine
from app.ml.explainability import Explainer

engine = RecommendationEngine(corpus, model)
for r in engine.recommend_for_case("2005_S_628", k=4).results:
    print(f"{r.hybrid_score:.3f}  {r.doc_id}  {r.title}  -- {r.rationale}")

ex = Explainer(corpus, model).explain_pair("2005_S_628", "2010_B_40")
print("\n", ex.summary)
print("contributions:", [(c.component, c.pct_of_score) for c in ex.contributions])
print("shared statutes:", [(s.act_name, s.section_number) for s in ex.shared_statutes])

# %% [markdown]
# ## 5. Evaluate

# %%
from app.ml.evaluation import evaluate_pairwise

for split in ("validation", "test"):
    m = evaluate_pairwise(lambda a, b: model.score(a, b), corpus.pairs_by_split[split])
    print(f"{split}: Spearman {m.spearman:+.3f}  Pearson {m.pearson:+.3f}  MSE {m.mse:.3f}")

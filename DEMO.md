# Demo walkthrough

## 0. One-time build (≈ 3 min)

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt
python scripts/build_all.py         # artifacts + DB
python scripts/evaluate.py          # writes data/processed/evaluation_report.json
cd frontend && npm install && cd ..
```

## 1. Start both servers

Two terminals (or `make demo` / `run.ps1`):

```bash
# terminal 1 — API
WARM_ON_STARTUP=1 uvicorn app.main:app --app-dir backend        # http://localhost:8000/docs

# terminal 2 — UI
cd frontend && npm run dev                                      # http://localhost:5173
```

`GET http://localhost:8000/api/health/ready` should report `"ready": true`.

## 2. The walkthrough (UI at :5173)

| # | Do | Point to make |
|---|---|---|
| 1 | Home → search **`medical negligence`** | Semantic search — none of the top hits have "medical" or "negligence" in the *title*; the model found them from content. |
| 2 | Open **Jacob Mathew v State of Punjab** → *Find similar cases* | Top results are *Kusum Sharma*, *Dr. P. B. Desai*, *Martin D'Souza* — the medical-negligence precedent line. Each row shows the hybrid / text / network score bars and a plain-English rationale. |
| 3 | On result #1 (Kusum Sharma) → **Why similar** | The explanation view: contribution chart (text ≈ 62 %, network ≈ 15 %, learned intercept ≈ 23 %); **shared statutes** IPC s.88 & s.92 with titles and ICF; **shared citations** — the classic English authorities *Bolam*, *Hunter v Hanley*, *Hucks v Cole*, *Maynard*, which both Indian judgments rely on. |
| 4 | Same page → **Read the two judgments side by side** | Both texts, with shared statute references (green) and shared cited-party names (amber) highlighted. |
| 5 | Back to the case → **Citation graph** | Hier-SPCNet neighbourhood: the case, the statutes it cites (ICF-ranked), and its most-similar cases; a section shared by two cases links to both. Click a blue node to jump. |
| 6 | Nav → **Upload** → drop a judgment PDF | Text-layer extraction (pages, char count; scanned PDFs are flagged, no OCR). Then *Find similar cases* runs the same pipeline on the uploaded document (network score = structural overlap only — it has no graph node, and the UI says so). |
| 7 | Nav → **Evaluation** | The full benchmark table + the amber box explaining why **test-split correlation ≈ 0 for every method, including trivial baselines** (range restriction; 31/90 pairs touch a graph-isolated case; even the clean subset is ≈ 0). |

## 3. What to say about the numbers (be honest)

- **Validation** (100 pairs): text TF-IDF Spearman **+0.76**; the learned hybrid matches
  it on rank correlation (**+0.75**) while roughly **halving MSE** (0.055 vs 0.119) —
  on this corpus the network mainly improves *calibration*, not re-ranking.
- We do **not** reproduce the paper's +11.8 % hybrid gain: their corpus has 53,210 cases
  and a dense citation graph; ours has 295 cases and **24 resolved case→case edges**
  (party names aren't resolvable to ids — documented, not a bug).
- **Test** (90 pairs): every method is ≈ 0. This is a property of that split, not the
  pipeline — see the Evaluation page and `EVALUATION.md`. The **ranking** sanity is still
  non-zero on test (Doc2Vec MRR 0.30) — the system still *retrieves* similar cases.
- This is an academic implementation of a published method. **Not legal advice.**

## 4. CLI equivalents (no UI needed)

```bash
python scripts/recommend.py --case 2005_S_628 --k 5 --explain 2010_B_40
python scripts/recommend.py --file path/to/your_case.pdf --k 10
python scripts/explain.py --query 2005_S_628 --target 2010_B_40 --json
python scripts/evaluate.py
```

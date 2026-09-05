# Project Progress

[✓] 0. Project Analysis
[✓] 1. Environment Setup        (.venv on Python 3.12, core deps installed, check_env.py passes)
[✓] 2. Repository Setup         (git init + first commit, README, pyproject, tests scaffold — 8 smoke tests pass)
[✓] 3. Dataset Setup            (loader + ingest_data.py + manifest; 295 uniques, integrity clean; 20 tests pass)
[✓] 4. Data Preprocessing       (preprocessing.py + preprocess.py; clean/sentences/tokens for 295 docs; 36 tests pass)
[✓] 5. PDF/Text Extraction      (text_extraction.py + extract_text.py; PDF/txt → shared preprocessing; selftest 99.8%; 47 tests pass)
[✓] 6. Citation Extraction      (citation_extraction.py + extract_citations.py; 24 case→case edges @90, 1.4% resolution — §3 limitation; 63 tests)
[✓] 7. Statute Extraction       (statute_extraction.py + curated sample hierarchy + extract_statutes.py; 12,324 refs, 297 acts (23 curated); 77 tests)
[✓] 8. Text Similarity          (text_similarity.py: TF-IDF/Doc2Vec/SBERT pluggable + index; generate_embeddings.py; 90 tests. NOTE: test-split text corr ~0 — flagged for Step 22)
[✓] 9. Network Construction     (network_model.py: PCNet 24 edges / Hier-SPCNet 3108 nodes 6725 edges + ICF; build_graph.py; 98 tests)
[✓] 10. Network Embedding       (network_embedding.py: node2vec + metapath2vec via Word2Vec on walks; generate_network_embeddings.py; 106 tests)
# --- MVP critical path (doing these first) ---
[✓] 11. Network Similarity      (network_similarity.py: embedding + ICF statute/citation overlap + breakdown; 116 tests)
[✓] 12. Hybrid Similarity       (hybrid_similarity.py: weighted-sum + learned/NN-Map-style combiner; build_hybrid.py; CV Spearman 0.75 val; 128 tests)
[✓] 13. Recommendation Engine   (recommendation.py: Top-K for case/uploaded-text + filters + rationale; recommend.py; medical-negligence line surfaces; 137 tests)
[✓] 14. Explainability          (explainability.py: SimilarityExplanation + contribution attribution + labelled shared statutes/citations; explain.py; 144 tests)
[✓] 17. Database                (config.py + database.py + models/ + repository.py + init_db.py; 295 cases loaded; 151 tests)
[✓] 18. FastAPI Backend         (main.py + RecommenderService + schemas + routers health/stats/cases/recommend/compare; build_all.py; 163 tests)
[✓] 20. Search                  (search.py: title/semantic/hybrid modes + facets; 169 tests)
[✓] 21. Upload Workflow         (upload.py: POST /upload + recommend/upload + compare/upload; 176 tests)
[✓] 19. React Frontend          (Vite+Tailwind+Router+Recharts; Home/Search/Case/Recommend/Compare/Upload; verified live in browser; build clean)
[✓] 22. Evaluation              (evaluation.py + evaluate.py + /api/evaluate + EVALUATION.md; val text ρ0.76 / learned-hybrid ρ0.75 MSE0.055; test ≈0 diagnosed; 184 tests)

# ---  MVP COMPLETE  ---

# --- optional / polish (after MVP) ---
[ ] 15. Case Comparison
[ ] 16. Citation Graph
[ ] 23. Testing
[ ] 24. Optimization
[ ] 25. Documentation
[ ] 26. Final Demo

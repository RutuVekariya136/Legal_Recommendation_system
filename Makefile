# Convenience targets. On Windows use Git Bash, or run the underlying commands
# directly / via run.ps1. PY defaults to the project venv.

PY ?= .venv/Scripts/python.exe

.PHONY: help install install-ml env build evaluate backend frontend frontend-install \
        test test-frontend test-all demo clean

help:
	@echo "install         create venv deps (backend/requirements.txt)"
	@echo "install-ml      + sentence-transformers / faiss-cpu"
	@echo "env             run scripts/check_env.py"
	@echo "build           scripts/build_all.py  (artifacts + DB)"
	@echo "evaluate        scripts/evaluate.py   (benchmark vs expert pairs)"
	@echo "backend         run the API (uvicorn, :8000)"
	@echo "frontend        run the UI (vite, :5173)"
	@echo "test            pytest (backend)"
	@echo "test-frontend   vitest"
	@echo "test-all        both"
	@echo "clean           remove generated artifacts + db (keeps data/raw)"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r backend/requirements.txt

install-ml:
	$(PY) -m pip install -r backend/requirements-ml.txt

env:
	$(PY) scripts/check_env.py

build:
	$(PY) scripts/build_all.py

evaluate:
	$(PY) scripts/evaluate.py

backend:
	WARM_ON_STARTUP=1 $(PY) -m uvicorn app.main:app --reload --app-dir backend

frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev

test:
	$(PY) -m pytest

test-frontend:
	cd frontend && npm run test

test-all: test test-frontend

demo: build evaluate
	@echo "Artifacts ready. Now run 'make backend' and 'make frontend' in two terminals."
	@echo "Then follow DEMO.md."

clean:
	rm -rf data/processed/* data/graphs/* data/embeddings/* legal_case_similarity.db
	@echo "Removed generated artifacts. Run 'make build' to regenerate."

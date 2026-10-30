"""Application settings (loaded from backend/.env, overridable by real env vars).

Keeps DB/data locations and model defaults in one place so switching SQLite ->
PostgreSQL is a `DATABASE_URL` change, not a code change (PROJECT_NOTES.md §4).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.paths import REPO_ROOT

_ENV_FILE = REPO_ROOT / "backend" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Legal Case Similarity API"
    app_env: str = "development"
    debug: bool = True

    database_url: str = "sqlite:///./legal_case_similarity.db"

    # "files" = the 296 base-paper docs from data/raw/
    # "db"    = every Case row (base + bulk-ingested, e.g. 33k ILDC corpus)
    corpus_source: str = "db"

    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_embeddings_dir: str = "data/embeddings"
    data_graphs_dir: str = "data/graphs"
    data_indexes_dir: str = "data/indexes"

    text_similarity_method: str = "tfidf"
    network_embedding_method: str = "metapath2vec"
    sbert_model_name: str = "all-MiniLM-L6-v2"

    # optional JSON file that extends resources/statute_hierarchy.json (P6)
    statute_hierarchy_path: str = ""

    hybrid_method: str = "learned"       # weighted | learned
    hybrid_alpha: float = 0.5
    hybrid_beta: float = 0.5
    default_top_k: int = 10

    cors_origins: str = "http://localhost:5173"

    upload_max_bytes: int = 20 * 1024 * 1024  # 20 MB

    # -- derived --------------------------------------------------
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def database_url_resolved(self) -> str:
        """Make a relative SQLite path absolute to the repo root."""
        prefix = "sqlite:///./"
        if self.database_url.startswith(prefix):
            target = (REPO_ROOT / self.database_url[len(prefix):]).as_posix()
            return f"sqlite:///{target}"
        return self.database_url

    @property
    def is_sqlite(self) -> bool:
        return self.database_url_resolved.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()

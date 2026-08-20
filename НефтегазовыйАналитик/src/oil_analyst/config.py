from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    tavily_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "oil_reports"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enable_dense: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranker: bool = True
    dense_top_k: int = Field(12, ge=1)
    sparse_top_k: int = Field(12, ge=1)
    final_top_k: int = Field(6, ge=1)
    min_rag_score: float = Field(0.0, ge=0, le=1)
    chunk_size: int = Field(900, ge=100)
    chunk_overlap: int = Field(120, ge=0)
    forecast_csv: str = "data/oil_prices.csv"
    reports_dir: str = "data/reports"
    prices_dir: str = "data/prices"
    web_preferred_domains: str = "opec.org,iea.org,eia.gov,reuters.com"

    @property
    def preferred_domains(self) -> list[str]:
        return [x.strip().lower() for x in self.web_preferred_domains.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

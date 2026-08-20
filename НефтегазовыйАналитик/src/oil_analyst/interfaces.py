from typing import Protocol, Sequence

from .models import MetadataFilters, RetrievedDocument, WebResult


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, documents: Sequence[RetrievedDocument], vectors: Sequence[list[float]]) -> None: ...
    def search(self, vector: list[float], limit: int,
               filters: MetadataFilters | None = None) -> list[RetrievedDocument]: ...


class WebSearchProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[WebResult]: ...


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...

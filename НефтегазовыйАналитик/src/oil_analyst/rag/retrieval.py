import math
import re
from collections import Counter
from typing import Callable, Sequence

from ..interfaces import EmbeddingProvider, VectorStore
from ..models import MetadataFilters, RetrievedDocument


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[\w'-]+", text.lower())
    translations = {
        "нефть": "oil", "нефти": "oil", "нефтяной": "oil", "спрос": "demand",
        "спросу": "demand", "мировой": "world", "мировому": "world",
        "предложение": "supply", "предложении": "supply", "предложения": "supply",
        "добыча": "production", "добыче": "production",
        "запасы": "inventories", "запасов": "inventories", "прогноз": "outlook",
        "ожидает": "outlook", "цена": "price", "цены": "price",
    }
    return tokens + [translations[token] for token in tokens if token in translations]


class BM25Index:
    """Small dependency-free BM25 index; suitable for report corpora used by the MVP."""
    def __init__(self, documents: Sequence[RetrievedDocument]):
        self.documents = list(documents)
        self.tokens = [_tokens(d.text) for d in documents]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        self.df = Counter(t for row in self.tokens for t in set(row))

    def search(self, query: str, limit: int,
               filters: MetadataFilters | None = None) -> list[RetrievedDocument]:
        scores = []
        n, k1, b = len(self.documents), 1.5, 0.75
        for doc, words in zip(self.documents, self.tokens):
            if filters and filters.active():
                metadata = doc.metadata
                if filters.organization and metadata.organization != filters.organization:
                    continue
                if filters.report_date and (not metadata.report_date or metadata.report_date.strftime("%Y-%m") != filters.report_date):
                    continue
                if filters.report_name and metadata.report_name != filters.report_name:
                    continue
            tf, score = Counter(words), 0.0
            for term in _tokens(query):
                freq = tf[term]
                if not freq: continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(words) / max(self.avgdl, 1)))
            scores.append(doc.model_copy(update={"score": max(score, 0)}))
        return sorted(scores, key=lambda d: d.score, reverse=True)[:limit]


class HybridRetriever:
    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore, sparse: BM25Index,
                 dense_k: int = 12, sparse_k: int = 12, final_k: int = 6,
                 reranker: Callable[[str, Sequence[RetrievedDocument]], Sequence[float]] | None = None):
        self.embeddings, self.store, self.sparse = embeddings, store, sparse
        self.dense_k, self.sparse_k, self.final_k, self.reranker = dense_k, sparse_k, final_k, reranker

    def retrieve(self, query: str, filters: MetadataFilters | None = None) -> list[RetrievedDocument]:
        vector = self.embeddings.embed_query(query)
        try:
            dense = self.store.search(vector, self.dense_k, filters)
        except TypeError:
            dense = self.store.search(vector, self.dense_k)
        sparse = self.sparse.search(query, self.sparse_k, filters)
        return self.fuse_and_rerank(query, dense, sparse)

    def fuse_and_rerank(self, query: str, dense: Sequence[RetrievedDocument],
                        sparse: Sequence[RetrievedDocument]) -> list[RetrievedDocument]:
        fused: dict[str, tuple[RetrievedDocument, float]] = {}
        for results in (dense, sparse):
            for rank, doc in enumerate(results):
                key = doc.metadata.chunk_id
                previous = fused.get(key, (doc, 0.0))[1]
                fused[key] = (doc, previous + 1 / (60 + rank + 1))
        ranked = [doc.model_copy(update={"score": score}) for doc, score in sorted(fused.values(), key=lambda x: x[1], reverse=True)]
        if self.reranker and ranked:
            ranked = [d.model_copy(update={"score": float(s)}) for d, s in zip(ranked, self.reranker(query, ranked))]
            ranked.sort(key=lambda d: d.score, reverse=True)
        query_tokens = set(_tokens(query))
        requested_section = next((name for name in ("supply", "demand") if name in query_tokens), None)
        if requested_section:
            ranked.sort(
                key=lambda d: (
                    requested_section in (d.metadata.section_title or "").casefold(),
                    d.score,
                ),
                reverse=True,
            )
        return ranked[:self.final_k]

from datetime import date
import logging
import time
from typing import Sequence
from urllib.parse import urlparse
import uuid

from .models import MetadataFilters, RetrievedDocument, WebResult

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, api_key: str | None = None):
        from qdrant_client import QdrantClient
        self.client, self.collection = QdrantClient(url=url, api_key=api_key), collection

    def upsert(self, documents: Sequence[RetrievedDocument], vectors: Sequence[list[float]]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams
        if not vectors: return
        collections = {x.name for x in self.client.get_collections().collections}
        if self.collection not in collections:
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, d.metadata.chunk_id)), vector=v,
                              payload=d.model_dump(mode="json")) for d, v in zip(documents, vectors)]
        self.client.upsert(self.collection, points=points)

    def search(self, vector: list[float], limit: int,
               filters: MetadataFilters | None = None) -> list[RetrievedDocument]:
        try:
            query_filter = None
            if filters and filters.active():
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                conditions = []
                if filters.organization:
                    conditions.append(FieldCondition(key="metadata.organization", match=MatchValue(value=filters.organization)))
                if filters.report_date:
                    conditions.append(FieldCondition(key="metadata.report_date", match=MatchValue(value=filters.report_date + "-01")))
                if filters.report_name:
                    conditions.append(FieldCondition(key="metadata.report_name", match=MatchValue(value=filters.report_name)))
                query_filter = Filter(must=conditions)
            result = self.client.query_points(self.collection, query=vector, limit=limit,
                                              query_filter=query_filter).points
        except Exception as exc:
            logger.warning("Qdrant search unavailable for collection %s: %s", self.collection, exc)
            return []
        return [RetrievedDocument.model_validate({**p.payload, "score": max(float(p.score), 0)}) for p in result]


class TavilySearch:
    def __init__(self, api_key: str, preferred_domains: list[str], timeout: float = 20.0,
                 max_attempts: int = 3, backoff_seconds: float = 0.5):
        from tavily import TavilyClient
        self.client, self.domains = TavilyClient(api_key), preferred_domains
        self.timeout, self.max_attempts, self.backoff_seconds = timeout, max_attempts, backoff_seconds

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                raw = self.client.search(query, max_results=limit, include_domains=self.domains,
                                         search_depth="advanced", timeout=self.timeout)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self.max_attempts:
                    raise
                time.sleep(self.backoff_seconds * (2 ** attempt))
        else:
            raise RuntimeError("Tavily search has no configured attempts") from last_error
        results = []
        for item in raw.get("results", []):
            host = urlparse(item["url"]).hostname or ""
            if not any(host == d or host.endswith("." + d) for d in self.domains): continue
            published = item.get("published_date")
            results.append(WebResult(title=item["title"], url=item["url"], content=item.get("content", ""), published_date=date.fromisoformat(published[:10]) if published else None, score=item.get("score", 0)))
        return results


class OpenAIGenerator:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key), model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt, temperature=0.1, max_output_tokens=1800)
        return response.output_text


class OllamaGenerator:
    """Minimal Ollama chat adapter requesting JSON structured output."""
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def generate(self, prompt: str) -> str:
        import httpx
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "uncertainty": {"type": ["string", "null"]},
            },
            "required": ["answer", "uncertainty"],
            "additionalProperties": False,
        }
        payload = {"model": self.model, "stream": False, "format": schema,
                   "options": {"temperature": 0.1},
                   "messages": [{"role": "system", "content": "Return only valid JSON matching the schema."},
                                {"role": "user", "content": prompt}]}
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def __call__(self, query: str, documents: Sequence[RetrievedDocument]) -> list[float]:
        return [float(score) for score in self.model.predict([(query, doc.text) for doc in documents])]

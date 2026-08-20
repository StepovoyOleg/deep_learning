from pathlib import Path
import logging

from .adapters import CrossEncoderReranker, OllamaGenerator, OpenAIGenerator, QdrantVectorStore, SentenceTransformerEmbeddings, TavilySearch
from .config import Settings, get_settings
from .forecasting import EiaBrentDataProvider
from .orchestration import AnalystAgent
from .rag.retrieval import BM25Index, HybridRetriever
from .rag.ingestion import PdfIngestor

logger = logging.getLogger(__name__)


def build_agent(settings: Settings | None = None) -> AnalystAgent:
    settings = settings or get_settings()
    # Parsing does not use these dependencies; placeholders keep the ingestion object lightweight.
    corpus = PdfIngestor(settings).load_directory(Path(settings.reports_dir))
    sparse = BM25Index(corpus)
    try:
        if not settings.enable_dense:
            raise RuntimeError("dense retrieval disabled by configuration")
        embeddings = SentenceTransformerEmbeddings(settings.embedding_model)
        store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key)
        try:
            reranker = CrossEncoderReranker(settings.reranker_model) if settings.enable_reranker else None
        except Exception as exc:
            logger.warning("Reranker unavailable; using RRF order: %s", exc)
            reranker = None
        hybrid = HybridRetriever(embeddings, store, sparse, settings.dense_top_k,
                                  settings.sparse_top_k, settings.final_top_k, reranker)
        retrieve = hybrid.retrieve
    except Exception as exc:
        logger.warning("Dense retrieval unavailable; using BM25 fallback: %s", exc)
        retrieve = lambda query, filters=None: sparse.search(query, settings.final_top_k, filters)
    web = TavilySearch(settings.tavily_api_key, settings.preferred_domains) if settings.tavily_api_key else None
    if settings.llm_provider.casefold() == "ollama":
        generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model)
    else:
        generator = OpenAIGenerator(settings.openai_api_key, settings.openai_model) if settings.openai_api_key else None
    price_file = next(Path(settings.prices_dir).glob("RBRTEd.xls"), Path(settings.forecast_csv))
    return AnalystAgent(retrieve, web, EiaBrentDataProvider(price_file), generator, settings.min_rag_score)

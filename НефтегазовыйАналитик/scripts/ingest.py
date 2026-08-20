import logging
from pathlib import Path

from oil_analyst.adapters import QdrantVectorStore, SentenceTransformerEmbeddings
from oil_analyst.config import get_settings
from oil_analyst.rag.ingestion import PdfIngestor

def main() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    try:
        embeddings = SentenceTransformerEmbeddings(settings.embedding_model)
        store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_api_key)
        stats = PdfIngestor(settings, embeddings, store).ingest_directory(Path(settings.reports_dir))
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1
    print(f"PDF files: {stats.pdf_files}")
    print(f"Pages processed: {stats.pages_processed}")
    print(f"Empty pages: {stats.empty_pages}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Vectors stored: {stats.vectors_stored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

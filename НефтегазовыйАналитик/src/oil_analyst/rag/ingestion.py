import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from ..config import Settings
from ..interfaces import EmbeddingProvider, VectorStore
from ..models import DocumentMetadata, RetrievedDocument

logger = logging.getLogger(__name__)

OPEC_SECTIONS = (
    "Crude Oil Price Movements", "Commodity Markets", "World Economy",
    "World Oil Demand", "World Oil Supply", "Product Markets and Refinery Operations",
    "Tanker Market", "Crude and Refined Products Trade", "Commercial Stock Movements",
    "Balance of Supply and Demand", "Appendix",
)


@dataclass(frozen=True)
class IngestionStats:
    pdf_files: int = 0
    pages_processed: int = 0
    empty_pages: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0


def infer_report_metadata(path: Path) -> tuple[str, str, date | None]:
    """Infer stable report metadata from the known OPEC/EIA file naming scheme."""
    name = path.stem.casefold()
    organization = "OPEC" if "opec" in name or path.parent.name.casefold() == "opec" else "EIA"
    report_name = "OPEC Monthly Oil Market Report" if organization == "OPEC" else "EIA Short-Term Energy Outlook"
    months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
              "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
    report_date = None
    long_match = re.search(r"(" + "|".join(months) + r")[-_ ](20\d{2})", name)
    short_match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2})", name)
    if long_match:
        report_date = date(int(long_match.group(2)), months[long_match.group(1)], 1)
    elif short_match:
        abbreviations = {month[:3]: number for month, number in months.items()}
        report_date = date(2000 + int(short_match.group(2)), abbreviations[short_match.group(1)], 1)
    return organization, report_name, report_date


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    result, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            end = boundary if boundary > start else end
        result.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [x for x in result if x]


def extract_page_context(raw_text: str, organization: str) -> tuple[str | None, list[str], list[str]]:
    """Extract only stable text-layer labels; deliberately avoids layout guessing."""
    lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]
    section = None
    if organization == "OPEC":
        prefix = " ".join(lines[:4]).casefold()
        section = next((title for title in OPEC_SECTIONS if title.casefold() in prefix), None)
    tables = [line for line in lines if re.match(r"^Table\s+(?:\d+\s*[-.]\s*\d+|\d+[a-z]?\.)", line, re.I)]
    graphs = [line for line in lines if re.match(r"^(?:Graph|Figure)\s+\d+", line, re.I)]
    return section, tables, graphs


def contextualize_chunk(chunk: str, organization: str, report_name: str, report_date: date | None,
                        page: int, section_title: str | None = None,
                        table_title: str | None = None) -> str:
    header = [f"Organization: {organization}", f"Report: {report_name}"]
    if report_date:
        header.append(f"Date: {report_date.strftime('%B %Y')}")
    if section_title:
        header.append(f"Section: {section_title}")
    header.append(f"Page: {page}")
    if table_title:
        header.append(f"Table: {table_title}")
    return "\n".join(header) + "\n\n" + chunk


class PdfIngestor:
    def __init__(self, settings: Settings, embeddings: EmbeddingProvider | None = None,
                 store: VectorStore | None = None):
        self.settings, self.embeddings, self.store = settings, embeddings, store

    def parse(self, path: Path) -> list[RetrievedDocument]:
        reader = PdfReader(path)
        organization, report_name, report_date = infer_report_metadata(path)
        title = (reader.metadata.title if reader.metadata else None) or report_name
        documents: list[RetrievedDocument] = []
        for page_number, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text() or ""
            section_title, table_titles, graph_titles = extract_page_context(raw_text, organization)
            for index, chunk in enumerate(chunk_text(clean_text(raw_text), self.settings.chunk_size, self.settings.chunk_overlap)):
                digest = hashlib.sha256(f"{path.name}:{page_number}:{index}:{chunk}".encode()).hexdigest()[:24]
                table_title = next((value for value in table_titles if value in chunk), None)
                graph_title = next((value for value in graph_titles if value in chunk), None)
                content_type = "table" if table_title else ("graph_caption" if graph_title else "text")
                metadata = DocumentMetadata(
                    document_name=path.name, document_title=title, organization=organization,
                    report_name=report_name, report_date=report_date, date=report_date,
                    source=path.as_posix(), page=page_number, chunk_id=digest,
                    section_title=section_title, table_title=table_title, content_type=content_type,
                )
                indexed_text = contextualize_chunk(chunk, organization, report_name, report_date,
                                                   page_number, section_title, table_title)
                documents.append(RetrievedDocument(text=indexed_text, metadata=metadata, score=0))
        return documents

    def load_directory(self, directory: Path) -> list[RetrievedDocument]:
        return [doc for path in sorted(directory.rglob("*.pdf")) for doc in self.parse(path)]

    def ingest_directory(self, directory: Path) -> IngestionStats:
        paths = sorted(directory.rglob("*.pdf"))
        pages_processed = empty_pages = 0
        documents: list[RetrievedDocument] = []
        for path in paths:
            reader = PdfReader(path)
            pages_processed += len(reader.pages)
            empty_pages += sum(not clean_text(page.extract_text() or "") for page in reader.pages)
            documents.extend(self.parse(path))
        if documents:
            if self.embeddings is None or self.store is None:
                raise RuntimeError("Embedding provider and vector store are required for ingestion")
            self.store.upsert(documents, self.embeddings.embed_documents([d.text for d in documents]))
        logger.info("Ingested %d chunks", len(documents))
        return IngestionStats(len(paths), pages_processed, empty_pages, len(documents), len(documents))

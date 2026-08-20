from datetime import date

import pytest

from oil_analyst.models import DocumentMetadata, RetrievedDocument


@pytest.fixture
def make_document():
    def factory(text: str, chunk_id: str = "c1", page: int = 1, score: float = 0.0):
        return RetrievedDocument(
            text=text,
            metadata=DocumentMetadata(
                document_name="opec.pdf", document_title="OPEC MOMR",
                organization="OPEC", report_name="OPEC Monthly Oil Market Report",
                report_date=date(2026, 7, 1), date=date(2026, 7, 1),
                source="data/reports/opec/opec.pdf", page=page, chunk_id=chunk_id,
            ),
            score=score,
        )
    return factory

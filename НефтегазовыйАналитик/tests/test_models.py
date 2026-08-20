from datetime import date

from oil_analyst.models import DocumentMetadata


def test_document_metadata_accepts_date_without_name_collision():
    metadata = DocumentMetadata(document_name="x.pdf", organization="EIA", report_name="STEO",
        report_date=date(2026, 7, 1), source="data/reports/eia/x.pdf", page=1, chunk_id="x")
    assert metadata.report_date == date(2026, 7, 1)

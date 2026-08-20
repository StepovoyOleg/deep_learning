from oil_analyst.models import MetadataFilters
from oil_analyst.rag.retrieval import BM25Index
from oil_analyst.routing import extract_metadata_filters


def test_extracts_opec_july_2026():
    filters = extract_metadata_filters("Что OPEC писал в июльском отчёте 2026 года?")
    assert filters.organization == "OPEC" and filters.report_date == "2026-07"


def test_extracts_eia_month_and_year():
    filters = extract_metadata_filters("EIA STEO June 2026 oil inventories")
    assert filters.organization == "EIA" and filters.report_date == "2026-06"
    assert filters.report_name == "EIA Short-Term Energy Outlook"


def test_query_without_constraints_has_no_filters():
    assert not extract_metadata_filters("global oil demand outlook").active()


def test_bm25_filters_before_ranking(make_document):
    opec = make_document("oil demand", "opec")
    eia = make_document("oil demand", "eia").model_copy(deep=True)
    eia.metadata.organization = "EIA"
    result = BM25Index([opec, eia]).search("oil demand", 5, MetadataFilters(organization="EIA"))
    assert [doc.metadata.organization for doc in result] == ["EIA"]

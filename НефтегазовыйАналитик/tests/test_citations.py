from oil_analyst.citations import report_source


def test_report_citation_comes_from_metadata(make_document):
    citation = report_source(make_document("text", page=31)).citation
    assert "OPEC Monthly Oil Market Report" in citation and "p. 31" in citation

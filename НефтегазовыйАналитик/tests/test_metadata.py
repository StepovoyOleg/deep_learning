from pathlib import Path

from oil_analyst.rag.ingestion import infer_report_metadata


def test_opec_filename_metadata():
    organization, name, stamp = infer_report_metadata(Path("data/reports/opec/opec-momr-july-2026.pdf"))
    assert (organization, name, stamp.isoformat()) == ("OPEC", "OPEC Monthly Oil Market Report", "2026-07-01")


def test_eia_filename_metadata():
    organization, _, stamp = infer_report_metadata(Path("data/reports/eia/jun26.pdf"))
    assert organization == "EIA" and stamp.isoformat() == "2026-06-01"

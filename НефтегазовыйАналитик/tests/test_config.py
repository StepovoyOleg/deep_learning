from oil_analyst.config import Settings


def test_retrieval_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("FINAL_TOP_K", "4")
    assert Settings(_env_file=None).final_top_k == 4


def test_invalid_overlap_is_rejected_by_chunker():
    from oil_analyst.rag.ingestion import chunk_text
    try: chunk_text("abc", 100, 100)
    except ValueError: pass
    else: raise AssertionError("expected ValueError")

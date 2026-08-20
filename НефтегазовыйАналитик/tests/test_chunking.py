from datetime import date

from oil_analyst.rag.ingestion import chunk_text, clean_text, contextualize_chunk, extract_page_context


def test_character_chunking_preserves_overlap_and_size():
    chunks = chunk_text("word " * 300, 100, 20)
    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


def test_clean_text_collapses_whitespace():
    assert clean_text("a\n\t b\x00c") == "a b c"


def test_opec_section_and_table_context_are_preserved():
    raw = """World Oil Supply
53 OPEC Monthly Oil Market Report – July 2026
DoC NGLs and non-conventional liquids
Table 5 - 6: DoC NGLs + non-conventional liquids production, mb/d
Total 8.63 8.76 8.87
"""
    section, tables, graphs = extract_page_context(raw, "OPEC")
    assert section == "World Oil Supply"
    assert tables == ["Table 5 - 6: DoC NGLs + non-conventional liquids production, mb/d"]
    assert not graphs
    original = "The 2026 forecast indicates an increase of 0.1 mb/d."
    enriched = contextualize_chunk(original, "OPEC", "OPEC Monthly Oil Market Report",
                                   date(2026, 7, 1), 53, section, tables[0])
    assert "Section: World Oil Supply" in enriched
    assert "Table: Table 5 - 6" in enriched
    assert enriched.endswith(original)
